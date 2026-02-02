import { useRef, useEffect, useMemo, useState } from 'react';
import { useLoader } from '@react-three/fiber';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import * as THREE from 'three';

/**
 * DifferenceMesh Component
 * 
 * Renders a "What-If" heatmap visualization showing exactly which parts
 * of the artery changed during the healing process.
 * 
 * Color Gradient:
 * - BLUE: No change (d = 0) - healthy parts that stayed the same
 * - RED: High change (d > threshold) - aneurysm bulge that shrank
 */
export default function DifferenceMesh({ 
  originalUrl, 
  healedUrl, 
  maxDistance = 0.2,
  visible = true 
}) {
  const meshRef = useRef();
  const [coloredGeometry, setColoredGeometry] = useState(null);
  
  // Load both OBJ files
  const originalObj = useLoader(OBJLoader, originalUrl);
  const healedObj = useLoader(OBJLoader, healedUrl);
  
  // Clone objects to avoid mutations
  const clonedOriginal = useMemo(() => originalObj.clone(true), [originalObj]);
  const clonedHealed = useMemo(() => healedObj.clone(true), [healedObj]);
  
  useEffect(() => {
    if (!clonedOriginal || !clonedHealed || !visible) return;
    
    console.log('DifferenceMesh: Starting heatmap computation...');
    
    try {
      // Extract geometry and vertices from original mesh
      let originalGeometry = null;
      let originalPositions = null;
      
      clonedOriginal.traverse((child) => {
        if (child.isMesh && child.geometry && !originalGeometry) {
          originalGeometry = child.geometry.clone();
          originalPositions = child.geometry.attributes.position;
        }
      });
      
      // Extract vertices from healed point cloud/mesh
      let healedPositions = null;
      clonedHealed.traverse((child) => {
        if (child.geometry && !healedPositions) {
          healedPositions = child.geometry.attributes.position;
        }
      });
      
      if (!originalGeometry || !originalPositions) {
        console.error('DifferenceMesh: No geometry in original mesh');
        return;
      }
      
      if (!healedPositions) {
        console.error('DifferenceMesh: No positions in healed mesh');
        return;
      }
      
      const origCount = originalPositions.count;
      const healedCount = healedPositions.count;
      console.log(`DifferenceMesh: Original: ${origCount}, Healed: ${healedCount} vertices`);
      
      // Build arrays for faster access
      const origVerts = [];
      for (let i = 0; i < origCount; i++) {
        origVerts.push({
          x: originalPositions.getX(i),
          y: originalPositions.getY(i),
          z: originalPositions.getZ(i)
        });
      }
      
      const healedVerts = [];
      for (let i = 0; i < healedCount; i++) {
        healedVerts.push({
          x: healedPositions.getX(i),
          y: healedPositions.getY(i),
          z: healedPositions.getZ(i)
        });
      }
      
      // Normalize both point sets
      const normalize = (verts) => {
        let cx = 0, cy = 0, cz = 0;
        for (const v of verts) { cx += v.x; cy += v.y; cz += v.z; }
        cx /= verts.length; cy /= verts.length; cz /= verts.length;
        
        let maxDist = 0;
        for (const v of verts) {
          const dx = v.x - cx, dy = v.y - cy, dz = v.z - cz;
          const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
          if (d > maxDist) maxDist = d;
        }
        
        const normalized = [];
        for (const v of verts) {
          normalized.push({
            x: (v.x - cx) / maxDist,
            y: (v.y - cy) / maxDist,
            z: (v.z - cz) / maxDist
          });
        }
        return normalized;
      };
      
      const origNorm = normalize(origVerts);
      const healedNorm = normalize(healedVerts);
      
      // Build a simple spatial hash for healed vertices (grid-based)
      const GRID_SIZE = 0.1;
      const spatialHash = new Map();
      
      const getKey = (x, y, z) => {
        const gx = Math.floor(x / GRID_SIZE);
        const gy = Math.floor(y / GRID_SIZE);
        const gz = Math.floor(z / GRID_SIZE);
        return `${gx},${gy},${gz}`;
      };
      
      for (let i = 0; i < healedNorm.length; i++) {
        const v = healedNorm[i];
        const key = getKey(v.x, v.y, v.z);
        if (!spatialHash.has(key)) {
          spatialHash.set(key, []);
        }
        spatialHash.get(key).push(v);
      }
      
      // Find nearest neighbor using spatial hash
      const findNearest = (v) => {
        const gx = Math.floor(v.x / GRID_SIZE);
        const gy = Math.floor(v.y / GRID_SIZE);
        const gz = Math.floor(v.z / GRID_SIZE);
        
        let minDist = Infinity;
        
        // Check neighboring cells (3x3x3 neighborhood)
        for (let dx = -1; dx <= 1; dx++) {
          for (let dy = -1; dy <= 1; dy++) {
            for (let dz = -1; dz <= 1; dz++) {
              const key = `${gx+dx},${gy+dy},${gz+dz}`;
              const cell = spatialHash.get(key);
              if (cell) {
                for (const h of cell) {
                  const distSq = (v.x - h.x) ** 2 + (v.y - h.y) ** 2 + (v.z - h.z) ** 2;
                  if (distSq < minDist) minDist = distSq;
                }
              }
            }
          }
        }
        
        // If no neighbors found in grid, expand search
        if (minDist === Infinity) {
          for (let i = 0; i < healedNorm.length; i++) {
            const h = healedNorm[i];
            const distSq = (v.x - h.x) ** 2 + (v.y - h.y) ** 2 + (v.z - h.z) ** 2;
            if (distSq < minDist) minDist = distSq;
          }
        }
        
        return Math.sqrt(minDist);
      };
      
      // Calculate distances for all original vertices
      const distances = [];
      for (let i = 0; i < origNorm.length; i++) {
        distances.push(findNearest(origNorm[i]));
      }
      
      // Get max distance for scaling
      let actualMax = 0;
      for (const d of distances) {
        if (d > actualMax) actualMax = d;
      }
      const colorScale = Math.max(maxDistance, actualMax * 0.5);
      
      console.log(`DifferenceMesh: Max distance: ${actualMax.toFixed(4)}, Scale: ${colorScale.toFixed(4)}`);
      
      // Create vertex colors
      const colors = new Float32Array(origCount * 3);
      
      for (let i = 0; i < distances.length; i++) {
        const t = Math.min(distances[i] / colorScale, 1.0);
        
        // Blue -> Cyan -> Green -> Yellow -> Red
        let r, g, b;
        if (t < 0.25) {
          const s = t / 0.25;
          r = 0; g = s; b = 1;
        } else if (t < 0.5) {
          const s = (t - 0.25) / 0.25;
          r = 0; g = 1; b = 1 - s;
        } else if (t < 0.75) {
          const s = (t - 0.5) / 0.25;
          r = s; g = 1; b = 0;
        } else {
          const s = (t - 0.75) / 0.25;
          r = 1; g = 1 - s; b = 0;
        }
        
        colors[i * 3] = r;
        colors[i * 3 + 1] = g;
        colors[i * 3 + 2] = b;
      }
      
      // Apply colors to geometry
      originalGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      
      // Center and scale
      originalGeometry.computeBoundingBox();
      const box = originalGeometry.boundingBox;
      const center = new THREE.Vector3();
      box.getCenter(center);
      originalGeometry.translate(-center.x, -center.y, -center.z);
      
      const size = new THREE.Vector3();
      box.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z);
      const scale = 2 / maxDim;
      originalGeometry.scale(scale, scale, scale);
      
      console.log('DifferenceMesh: Heatmap geometry created successfully!');
      setColoredGeometry(originalGeometry);
      
    } catch (err) {
      console.error('DifferenceMesh: Error computing heatmap:', err);
    }
    
  }, [clonedOriginal, clonedHealed, maxDistance, visible]);
  
  if (!coloredGeometry || !visible) return null;
  
  return (
    <mesh ref={meshRef} geometry={coloredGeometry}>
      <meshStandardMaterial
        vertexColors={true}
        side={THREE.DoubleSide}
        metalness={0.2}
        roughness={0.5}
      />
    </mesh>
  );
}

/**
 * HeatmapLegend Component
 */
export function HeatmapLegend() {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <div className="w-24 h-4 rounded" 
          style={{ 
            background: 'linear-gradient(to right, #3b82f6, #06b6d4, #10b981, #eab308, #f97316, #ef4444)' 
          }} 
        />
      </div>
      <div className="flex justify-between text-xs text-slate-400">
        <span>No Change</span>
        <span>High Change</span>
      </div>
      <p className="text-xs text-slate-500 italic">
        Blue = healthy tissue | Red = aneurysm site
      </p>
    </div>
  );
}
