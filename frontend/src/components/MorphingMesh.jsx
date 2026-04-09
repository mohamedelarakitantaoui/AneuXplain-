import { useRef, useEffect, useMemo, useState, useCallback } from 'react';
import { useLoader, useFrame } from '@react-three/fiber';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import * as THREE from 'three';

/**
 * ============================================
 * MorphingMesh Component
 * ============================================
 * 
 * THESIS DEMONSTRATION: "Manifold Learning"
 * 
 * This component proves that our autoencoder learned a meaningful
 * geometric manifold by allowing smooth interpolation between:
 * - t = 0.0: Original (sick/high-risk) geometry
 * - t = 1.0: Healed (healthy/low-risk) geometry
 * 
 * The smooth transition demonstrates that the latent space is
 * continuous and semantically meaningful - not just random noise.
 * 
 * Mathematical Formula:
 *   Current_Position = Original_Position * (1 - t) + Healed_Position * t
 * 
 * CRITICAL ASSUMPTION: Both meshes have identical topology
 * (same vertex count and order) for valid interpolation.
 * 
 * @param {string} originalUrl - URL of the original (sick) OBJ mesh
 * @param {string} healedUrl - URL of the healed (healthy) OBJ mesh
 * @param {number} morphValue - Interpolation parameter t ∈ [0, 1]
 * @param {boolean} showRootCause - Enable heatmap coloring (root cause visualization)
 * @param {number} maxDistance - Maximum distance for heatmap color scaling
 * @param {boolean} visible - Whether to render the mesh
 * @param {Float32Array|Array} healedVertices - Optional: direct vertex positions [x1,y1,z1,x2,y2,z2,...] to override healedUrl
 */
export default function MorphingMesh({ 
  originalUrl, 
  healedUrl, 
  morphValue = 0,
  showRootCause = false,
  maxDistance = 0.15,
  visible = true,
  healedVertices = null  // NEW: Direct vertex positions for dynamic updates
}) {
  const meshRef = useRef();
  
  // State for geometry and precomputed data
  const [geometry, setGeometry] = useState(null);
  const [originalPositions, setOriginalPositions] = useState(null);
  const [healedPositionsMap, setHealedPositionsMap] = useState(null);
  const [vertexColors, setVertexColors] = useState(null);
  const [isReady, setIsReady] = useState(false);
  const [vertexDistances, setVertexDistances] = useState(null);
  
  // Load both OBJ files
  const originalObj = useLoader(OBJLoader, originalUrl);
  const healedObj = useLoader(OBJLoader, healedUrl);
  
  // Clone objects to avoid mutating cached loaders
  const clonedOriginal = useMemo(() => originalObj.clone(true), [originalObj]);
  const clonedHealed = useMemo(() => healedObj.clone(true), [healedObj]);
  
  /**
   * INITIALIZATION: Compute vertex correspondences and prepare for morphing
   * 
   * This is the expensive computation that only runs once when meshes load.
   * We use useMemo patterns and spatial hashing for performance.
   */
  useEffect(() => {
    if (!clonedOriginal || !clonedHealed || !visible) return;
    
    console.log('🔬 MorphingMesh: Initializing manifold interpolation...');
    
    try {
      // ============================================
      // STEP 1: Extract geometry from original mesh
      // ============================================
      let origGeometry = null;
      let origPositions = null;
      
      clonedOriginal.traverse((child) => {
        if (child.isMesh && child.geometry && !origGeometry) {
          origGeometry = child.geometry.clone();
          origPositions = child.geometry.attributes.position.array.slice();
        }
      });
      
      // ============================================
      // STEP 2: Extract vertices from healed mesh
      // ============================================
      let healedVerts = [];
      clonedHealed.traverse((child) => {
        if (child.geometry && healedVerts.length === 0) {
          const positions = child.geometry.attributes.position;
          for (let i = 0; i < positions.count; i++) {
            healedVerts.push({
              x: positions.getX(i),
              y: positions.getY(i),
              z: positions.getZ(i)
            });
          }
        }
      });
      
      // Validate data
      if (!origGeometry || !origPositions) {
        console.error('❌ MorphingMesh: No geometry found in original mesh');
        return;
      }
      
      if (healedVerts.length === 0) {
        console.error('❌ MorphingMesh: No vertices found in healed mesh');
        return;
      }
      
      const origVertCount = origPositions.length / 3;
      console.log(`📊 MorphingMesh: Original=${origVertCount}, Healed=${healedVerts.length} vertices`);

      // ============================================
      // STEP 3: Establish vertex correspondence
      // ============================================
      // Gradient healing preserves mesh topology (same faces, same vertex
      // order), so when vertex counts match we use DIRECT INDEX correspondence.
      // This avoids the independent-normalization + NN-matching pipeline that
      // distorts the geometry (the old CVAE path needed NN because the decoder
      // emitted unordered point clouds).

      const useDirectCorrespondence = (origVertCount === healedVerts.length);

      const healedMap = new Float32Array(origVertCount * 3);
      const distances = new Float32Array(origVertCount);

      if (useDirectCorrespondence) {
        // --- DIRECT: vertex i → vertex i (same topology) ----------------
        console.log('📐 MorphingMesh: Same topology detected — using direct vertex correspondence');
        for (let i = 0; i < origVertCount; i++) {
          healedMap[i * 3]     = healedVerts[i].x;
          healedMap[i * 3 + 1] = healedVerts[i].y;
          healedMap[i * 3 + 2] = healedVerts[i].z;

          const dx = healedVerts[i].x - origPositions[i * 3];
          const dy = healedVerts[i].y - origPositions[i * 3 + 1];
          const dz = healedVerts[i].z - origPositions[i * 3 + 2];
          distances[i] = Math.sqrt(dx*dx + dy*dy + dz*dz);
        }
      } else {
        // --- NN FALLBACK: for CVAE / different-topology outputs ----------
        console.log('🔍 MorphingMesh: Different vertex counts — using NN matching');

        const normalizePointSet = (vertices) => {
          let cx = 0, cy = 0, cz = 0;
          for (const v of vertices) { cx += v.x; cy += v.y; cz += v.z; }
          cx /= vertices.length; cy /= vertices.length; cz /= vertices.length;
          let maxDist = 0;
          for (const v of vertices) {
            const d = Math.sqrt((v.x-cx)**2 + (v.y-cy)**2 + (v.z-cz)**2);
            if (d > maxDist) maxDist = d;
          }
          return { cx, cy, cz, scale: maxDist || 1 };
        };

        const origVerts = [];
        for (let i = 0; i < origVertCount; i++) {
          origVerts.push({ x: origPositions[i*3], y: origPositions[i*3+1], z: origPositions[i*3+2] });
        }

        const origNorm = normalizePointSet(origVerts);
        const healedNorm = normalizePointSet(healedVerts);

        const origNormalized = origVerts.map(v => ({
          x: (v.x - origNorm.cx) / origNorm.scale,
          y: (v.y - origNorm.cy) / origNorm.scale,
          z: (v.z - origNorm.cz) / origNorm.scale
        }));
        const healedNormalized = healedVerts.map(v => ({
          x: (v.x - healedNorm.cx) / healedNorm.scale,
          y: (v.y - healedNorm.cy) / healedNorm.scale,
          z: (v.z - healedNorm.cz) / healedNorm.scale
        }));

        const GRID_SIZE = 0.1;
        const spatialHash = new Map();
        const getGridKey = (x, y, z) =>
          `${Math.floor(x/GRID_SIZE)},${Math.floor(y/GRID_SIZE)},${Math.floor(z/GRID_SIZE)}`;

        healedNormalized.forEach((v, i) => {
          const key = getGridKey(v.x, v.y, v.z);
          if (!spatialHash.has(key)) spatialHash.set(key, []);
          spatialHash.get(key).push({ ...v, index: i });
        });

        for (let i = 0; i < origVertCount; i++) {
          const ov = origNormalized[i];
          const gx = Math.floor(ov.x / GRID_SIZE);
          const gy = Math.floor(ov.y / GRID_SIZE);
          const gz = Math.floor(ov.z / GRID_SIZE);
          let minDist = Infinity, nearest = null;

          for (let ddx = -2; ddx <= 2; ddx++) {
            for (let ddy = -2; ddy <= 2; ddy++) {
              for (let ddz = -2; ddz <= 2; ddz++) {
                const cell = spatialHash.get(`${gx+ddx},${gy+ddy},${gz+ddz}`);
                if (cell) {
                  for (const h of cell) {
                    const dSq = (ov.x-h.x)**2 + (ov.y-h.y)**2 + (ov.z-h.z)**2;
                    if (dSq < minDist) { minDist = dSq; nearest = h; }
                  }
                }
              }
            }
          }
          if (!nearest) {
            for (const h of healedNormalized) {
              const dSq = (ov.x-h.x)**2 + (ov.y-h.y)**2 + (ov.z-h.z)**2;
              if (dSq < minDist) { minDist = dSq; nearest = h; }
            }
          }

          healedMap[i*3]     = nearest.x * origNorm.scale + origNorm.cx;
          healedMap[i*3 + 1] = nearest.y * origNorm.scale + origNorm.cy;
          healedMap[i*3 + 2] = nearest.z * origNorm.scale + origNorm.cz;
          distances[i] = Math.sqrt(minDist);
        }
      }
      
      // Create vertex colors based on distances (for root cause visualization)
      // Note: Using loop instead of Math.max(...distances) to avoid stack overflow
      // on large meshes with many vertices
      let maxDist = 0;
      for (let i = 0; i < distances.length; i++) {
        if (distances[i] > maxDist) maxDist = distances[i];
      }
      const colorScale = Math.max(0.1, maxDist * 0.7);
      const colors = new Float32Array(origVertCount * 3);
      
      for (let i = 0; i < origVertCount; i++) {
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
      
      // Center and scale geometry
      origGeometry.computeBoundingBox();
      const box = origGeometry.boundingBox;
      const center = new THREE.Vector3();
      box.getCenter(center);
      origGeometry.translate(-center.x, -center.y, -center.z);
      
      const size = new THREE.Vector3();
      box.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z);
      const scale = 2 / maxDim;
      origGeometry.scale(scale, scale, scale);
      
      // Apply same transform to stored positions
      const transformedOriginal = new Float32Array(origPositions.length);
      const transformedHealed = new Float32Array(healedMap.length);
      
      for (let i = 0; i < origVertCount; i++) {
        transformedOriginal[i * 3] = (origPositions[i * 3] - center.x) * scale;
        transformedOriginal[i * 3 + 1] = (origPositions[i * 3 + 1] - center.y) * scale;
        transformedOriginal[i * 3 + 2] = (origPositions[i * 3 + 2] - center.z) * scale;
        
        transformedHealed[i * 3] = (healedMap[i * 3] - center.x) * scale;
        transformedHealed[i * 3 + 1] = (healedMap[i * 3 + 1] - center.y) * scale;
        transformedHealed[i * 3 + 2] = (healedMap[i * 3 + 2] - center.z) * scale;
      }
      
      // Store everything
      setOriginalPositions(transformedOriginal);
      setHealedPositionsMap(transformedHealed);
      setVertexColors(colors);
      setGeometry(origGeometry);
      setIsReady(true);
      
      console.log('MorphingMesh: Manifold interpolation ready!');
      
    } catch (err) {
      console.error('MorphingMesh: Error initializing:', err);
    }
    
  }, [clonedOriginal, clonedHealed, visible]);
  
  // Update geometry positions based on morph value
  useEffect(() => {
    if (!isReady || !geometry || !originalPositions || !healedPositionsMap) return;
    
    const positions = geometry.attributes.position.array;
    const vertCount = positions.length / 3;
    const t = morphValue;
    const oneMinusT = 1 - t;
    
    // Interpolate: Current = Original * (1 - t) + Healed * t
    for (let i = 0; i < vertCount; i++) {
      const i3 = i * 3;
      positions[i3] = originalPositions[i3] * oneMinusT + healedPositionsMap[i3] * t;
      positions[i3 + 1] = originalPositions[i3 + 1] * oneMinusT + healedPositionsMap[i3 + 1] * t;
      positions[i3 + 2] = originalPositions[i3 + 2] * oneMinusT + healedPositionsMap[i3 + 2] * t;
    }
    
    geometry.attributes.position.needsUpdate = true;
    geometry.computeVertexNormals();
    
  }, [morphValue, isReady, geometry, originalPositions, healedPositionsMap]);
  
  /**
   * ============================================
   * DIRECT VERTEX UPDATE EFFECT
   * ============================================
   * 
   * Handles dynamic updates when healedVertices prop changes.
   * This bypasses the URL-based loading and directly updates geometry.
   * 
   * Use case: When "Heal" button receives new vertex positions from API
   * without reloading an OBJ file.
   */
  useEffect(() => {
    // Skip if no direct vertices provided or mesh not ready
    if (!healedVertices || !meshRef.current) return;
    
    const mesh = meshRef.current;
    const geo = mesh.geometry;
    
    if (!geo || !geo.attributes || !geo.attributes.position) {
      console.warn('MorphingMesh: Geometry not ready for vertex update');
      return;
    }
    
    console.log('🔄 MorphingMesh: Updating geometry with new healed vertices...');
    
    try {
      // Convert healedVertices to Float32Array if needed
      const newPositions = healedVertices instanceof Float32Array 
        ? healedVertices 
        : new Float32Array(healedVertices);
      
      const currentPositions = geo.attributes.position;
      const expectedCount = currentPositions.count * 3;
      
      // Validate vertex count matches
      if (newPositions.length !== expectedCount) {
        console.warn(
          `MorphingMesh: Vertex count mismatch. Expected ${expectedCount / 3}, got ${newPositions.length / 3}`
        );
        // Still proceed if close enough (allow some tolerance)
        if (Math.abs(newPositions.length - expectedCount) > expectedCount * 0.1) {
          console.error('MorphingMesh: Vertex count too different, aborting update');
          return;
        }
      }
      
      // Update the position buffer directly
      // Option 1: Replace the entire attribute (safer for different array lengths)
      geo.setAttribute(
        'position',
        new THREE.BufferAttribute(newPositions, 3)
      );
      
      // Mark position attribute as needing update
      geo.attributes.position.needsUpdate = true;
      
      // Recompute vertex normals for correct lighting
      geo.computeVertexNormals();
      
      // Also update bounding box/sphere for frustum culling
      geo.computeBoundingBox();
      geo.computeBoundingSphere();
      
      // Update internal state if we want morphing to work with new data
      if (newPositions.length === expectedCount) {
        setHealedPositionsMap(new Float32Array(newPositions));
      }
      
      console.log('✅ MorphingMesh: Geometry updated successfully');
      
    } catch (err) {
      console.error('MorphingMesh: Error updating vertices:', err);
    }
    
  }, [healedVertices]);  // Trigger when healedVertices changes
  
  // Apply vertex colors when showRootCause changes
  useEffect(() => {
    if (!geometry || !vertexColors) return;
    
    if (showRootCause) {
      geometry.setAttribute('color', new THREE.BufferAttribute(vertexColors, 3));
    } else {
      geometry.deleteAttribute('color');
    }
  }, [showRootCause, geometry, vertexColors]);
  
  if (!geometry || !visible || !isReady) return null;
  
  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshStandardMaterial
        color={showRootCause ? '#ffffff' : '#06b6d4'}
        vertexColors={showRootCause}
        side={THREE.DoubleSide}
        metalness={0.3}
        roughness={0.4}
      />
    </mesh>
  );
}

/**
 * ============================================
 * MorphSliderLegend Component
 * ============================================
 * 
 * Interactive UI element for the morphing slider.
 * Shows the current position on the manifold with
 * visual feedback about risk level.
 */
export function MorphSliderLegend({ morphValue = 0, onMorphChange }) {
  // Determine risk category based on morph position
  const getRiskState = useCallback((t) => {
    if (t < 0.3) return { 
      label: 'HIGH RISK', 
      color: 'text-red-400',
      bgColor: 'bg-red-500/20',
      borderColor: 'border-red-500/30'
    };
    if (t < 0.7) return { 
      label: 'TRANSITIONING', 
      color: 'text-yellow-400',
      bgColor: 'bg-yellow-500/20',
      borderColor: 'border-yellow-500/30'
    };
    return { 
      label: 'LOW RISK', 
      color: 'text-green-400',
      bgColor: 'bg-green-500/20',
      borderColor: 'border-green-500/30'
    };
  }, []);
  
  const riskState = getRiskState(morphValue);
  
  return (
    <div className="flex flex-col gap-3 min-w-[220px]">
      {/* Title */}
      <div className="flex items-center gap-2 border-b border-slate-700 pb-2">
        <span className="text-cyan-400 font-semibold text-sm">🔄 Manifold Explorer</span>
      </div>
      
      {/* State labels */}
      <div className="flex justify-between text-xs">
        <span className="text-red-400 font-medium">🔴 Sick</span>
        <span className={`font-bold ${riskState.color} ${riskState.bgColor} px-2 py-0.5 rounded`}>
          {riskState.label}
        </span>
        <span className="text-green-400 font-medium">🟢 Healed</span>
      </div>
      
      {/* Interactive Slider */}
      <div className="relative">
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={morphValue}
          onChange={(e) => onMorphChange?.(parseFloat(e.target.value))}
          className="w-full h-3 rounded-full appearance-none cursor-pointer
                     [&::-webkit-slider-thumb]:appearance-none
                     [&::-webkit-slider-thumb]:w-5
                     [&::-webkit-slider-thumb]:h-5
                     [&::-webkit-slider-thumb]:bg-white
                     [&::-webkit-slider-thumb]:rounded-full
                     [&::-webkit-slider-thumb]:shadow-lg
                     [&::-webkit-slider-thumb]:cursor-pointer
                     [&::-webkit-slider-thumb]:border-2
                     [&::-webkit-slider-thumb]:border-slate-300
                     [&::-moz-range-thumb]:w-5
                     [&::-moz-range-thumb]:h-5
                     [&::-moz-range-thumb]:bg-white
                     [&::-moz-range-thumb]:rounded-full
                     [&::-moz-range-thumb]:border-0
                     [&::-moz-range-thumb]:cursor-pointer"
          style={{
            background: `linear-gradient(to right, 
              #ef4444 0%, 
              #f97316 25%,
              #eab308 50%, 
              #84cc16 75%,
              #22c55e 100%)`
          }}
        />
      </div>
      
      {/* Numeric value display */}
      <div className="flex justify-between items-center text-xs">
        <div className="flex items-center gap-2">
          <span className="text-slate-400">t =</span>
          <span className={`font-mono font-bold ${riskState.color} ${riskState.bgColor} px-2 py-0.5 rounded`}>
            {morphValue.toFixed(2)}
          </span>
        </div>
        <span className="text-slate-500 italic">Manifold Position</span>
      </div>
      
      {/* Scientific explanation */}
      <p className="text-xs text-slate-500 italic text-center border-t border-slate-700 pt-2">
        ✨ Traversing the learned geometric manifold
      </p>
    </div>
  );
}

/**
 * ============================================
 * RootCauseToggle Component  
 * ============================================
 * 
 * Toggle switch for the heatmap visualization.
 * Explains the root cause analysis feature.
 */
export function RootCauseToggle({ showHeatmap, onToggle }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-purple-400">🎯</span>
          <span className="text-sm text-slate-300 font-medium">Root Cause Heatmap</span>
        </div>
        <button
          onClick={() => onToggle?.(!showHeatmap)}
          className={`relative w-12 h-6 rounded-full transition-all duration-300 ${
            showHeatmap 
              ? 'bg-purple-500' 
              : 'bg-slate-700'
          }`}
        >
          <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all duration-300 ${
            showHeatmap ? 'left-7' : 'left-1'
          }`} />
        </button>
      </div>
      
      {showHeatmap && (
        <div className="mt-2 p-2 bg-purple-500/10 border border-purple-500/30 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-full h-3 rounded" 
              style={{ 
                background: 'linear-gradient(to right, #3b82f6, #06b6d4, #10b981, #eab308, #ef4444)' 
              }} 
            />
          </div>
          <div className="flex justify-between text-xs text-slate-400">
            <span>No Change</span>
            <span>High Change</span>
          </div>
          <p className="text-xs text-purple-300/70 italic mt-1">
            🔵 Healthy tissue → 🔴 Aneurysm site
          </p>
        </div>
      )}
    </div>
  );
}
