import { useRef, useEffect, useMemo, useState } from 'react';
import { useLoader } from '@react-three/fiber';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import * as THREE from 'three';

/**
 * ============================================
 * MorphArtery Component
 * ============================================
 * 
 * A single mesh that smoothly morphs between "Baseline Anatomy" (original)
 * and "Optimized Geometry" (healed) using Three.js morph targets.
 * 
 * NO MORE rendering two meshes on top of each other!
 * Instead, we use GPU-accelerated morphTargetInfluences for smooth transitions.
 * 
 * @param {string} originalUrl - URL of the baseline (sick) OBJ mesh
 * @param {string} healedUrl - URL of the optimized (healed) OBJ mesh
 * @param {number} morphValue - Interpolation parameter t ∈ [0, 1]
 *                              0 = Baseline Anatomy, 1 = Optimized Geometry
 * @param {boolean} visible - Whether to render the mesh
 * @param {Array} clippingPlanes - Optional clipping planes for cross-section views
 */
export default function MorphArtery({ 
  originalUrl, 
  healedUrl, 
  morphValue = 0,
  visible = true,
  clippingPlanes = [],
}) {
  const meshRef = useRef();
  const [isReady, setIsReady] = useState(false);
  const [morphGeometry, setMorphGeometry] = useState(null);
  
  // Load both OBJ files
  const originalObj = useLoader(OBJLoader, originalUrl);
  const healedObj = useLoader(OBJLoader, healedUrl);
  
  // Clone to avoid mutating cached loaders
  const clonedOriginal = useMemo(() => originalObj.clone(true), [originalObj]);
  const clonedHealed = useMemo(() => healedObj.clone(true), [healedObj]);
  
  /**
   * Initialize the morph target geometry
   * This runs once when both meshes are loaded
   */
  useEffect(() => {
    if (!clonedOriginal || !clonedHealed || !visible) return;
    
    console.log('🔄 MorphArtery: Setting up morph targets...');
    
    try {
      // ============================================
      // STEP 1: Extract geometry from original mesh
      // ============================================
      let origGeometry = null;
      clonedOriginal.traverse((child) => {
        if (child.isMesh && child.geometry && !origGeometry) {
          origGeometry = child.geometry.clone();
        }
      });
      
      if (!origGeometry) {
        console.error('MorphArtery: No geometry in original mesh');
        return;
      }
      
      // ============================================
      // STEP 2: Extract vertices from healed mesh
      // ============================================
      let healedPositions = null;
      clonedHealed.traverse((child) => {
        if (child.geometry && !healedPositions) {
          healedPositions = child.geometry.attributes.position;
        }
      });
      
      if (!healedPositions) {
        console.error('MorphArtery: No positions in healed mesh');
        return;
      }
      
      const origPositions = origGeometry.attributes.position;
      const origCount = origPositions.count;
      const healedCount = healedPositions.count;
      
      console.log(`📊 MorphArtery: Original=${origCount}, Healed=${healedCount} vertices`);
      
      // ============================================
      // STEP 3: Normalize both geometries to same space
      // ============================================
      const normalizeGeometry = (geometry) => {
        geometry.computeBoundingBox();
        const box = geometry.boundingBox;
        const center = new THREE.Vector3();
        box.getCenter(center);
        geometry.translate(-center.x, -center.y, -center.z);
        
        const size = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 2 / maxDim;
        geometry.scale(scale, scale, scale);
        
        return { center, scale };
      };
      
      const origTransform = normalizeGeometry(origGeometry);
      
      // ============================================
      // STEP 4: Build morph target from healed positions
      // ============================================
      // If vertex counts match, use direct mapping
      // Otherwise, use nearest-neighbor correspondence
      
      let morphTargetPositions;
      
      if (origCount === healedCount) {
        // Direct mapping - vertices correspond 1:1
        console.log('✅ MorphArtery: Direct vertex mapping');
        
        // Normalize healed positions to same space as original
        const healedVerts = [];
        for (let i = 0; i < healedCount; i++) {
          healedVerts.push(new THREE.Vector3(
            healedPositions.getX(i),
            healedPositions.getY(i),
            healedPositions.getZ(i)
          ));
        }
        
        // Compute healed centroid and scale
        const healedCenter = new THREE.Vector3();
        for (const v of healedVerts) healedCenter.add(v);
        healedCenter.divideScalar(healedVerts.length);
        
        let maxDist = 0;
        for (const v of healedVerts) {
          const d = v.clone().sub(healedCenter).length();
          if (d > maxDist) maxDist = d;
        }
        const healedScale = 2 / maxDist;
        
        // Apply normalization
        morphTargetPositions = new Float32Array(origCount * 3);
        for (let i = 0; i < origCount; i++) {
          const v = healedVerts[i].clone().sub(healedCenter).multiplyScalar(healedScale);
          morphTargetPositions[i * 3] = v.x;
          morphTargetPositions[i * 3 + 1] = v.y;
          morphTargetPositions[i * 3 + 2] = v.z;
        }
        
      } else {
        // Different vertex counts - use nearest neighbor matching
        console.log('⚠️ MorphArtery: Using nearest-neighbor correspondence');
        
        // Build normalized healed vertex array
        const healedVerts = [];
        for (let i = 0; i < healedCount; i++) {
          healedVerts.push({
            x: healedPositions.getX(i),
            y: healedPositions.getY(i),
            z: healedPositions.getZ(i)
          });
        }
        
        // Normalize healed vertices
        let hcx = 0, hcy = 0, hcz = 0;
        for (const v of healedVerts) { hcx += v.x; hcy += v.y; hcz += v.z; }
        hcx /= healedVerts.length; hcy /= healedVerts.length; hcz /= healedVerts.length;
        
        let hMaxDist = 0;
        for (const v of healedVerts) {
          const d = Math.sqrt((v.x-hcx)**2 + (v.y-hcy)**2 + (v.z-hcz)**2);
          if (d > hMaxDist) hMaxDist = d;
        }
        const hScale = 2 / hMaxDist;
        
        const healedNormalized = healedVerts.map(v => ({
          x: (v.x - hcx) * hScale,
          y: (v.y - hcy) * hScale,
          z: (v.z - hcz) * hScale
        }));
        
        // Build spatial hash for fast lookup
        const GRID_SIZE = 0.1;
        const spatialHash = new Map();
        const getKey = (x, y, z) => `${Math.floor(x/GRID_SIZE)},${Math.floor(y/GRID_SIZE)},${Math.floor(z/GRID_SIZE)}`;
        
        healedNormalized.forEach((v, i) => {
          const key = getKey(v.x, v.y, v.z);
          if (!spatialHash.has(key)) spatialHash.set(key, []);
          spatialHash.get(key).push({ ...v, idx: i });
        });
        
        // Find nearest healed vertex for each original vertex
        morphTargetPositions = new Float32Array(origCount * 3);
        const origPosArray = origGeometry.attributes.position.array;
        
        for (let i = 0; i < origCount; i++) {
          const ox = origPosArray[i * 3];
          const oy = origPosArray[i * 3 + 1];
          const oz = origPosArray[i * 3 + 2];
          
          // Search nearby grid cells
          const gx = Math.floor(ox / GRID_SIZE);
          const gy = Math.floor(oy / GRID_SIZE);
          const gz = Math.floor(oz / GRID_SIZE);
          
          let minDist = Infinity;
          let nearest = { x: ox, y: oy, z: oz };
          
          for (let dx = -2; dx <= 2; dx++) {
            for (let dy = -2; dy <= 2; dy++) {
              for (let dz = -2; dz <= 2; dz++) {
                const key = `${gx+dx},${gy+dy},${gz+dz}`;
                const cell = spatialHash.get(key);
                if (cell) {
                  for (const h of cell) {
                    const dist = (ox-h.x)**2 + (oy-h.y)**2 + (oz-h.z)**2;
                    if (dist < minDist) {
                      minDist = dist;
                      nearest = h;
                    }
                  }
                }
              }
            }
          }
          
          morphTargetPositions[i * 3] = nearest.x;
          morphTargetPositions[i * 3 + 1] = nearest.y;
          morphTargetPositions[i * 3 + 2] = nearest.z;
        }
      }
      
      // ============================================
      // STEP 5: Create morph target attribute
      // ============================================
      // The morph target stores the DIFFERENCE from the base position
      const morphDelta = new Float32Array(origCount * 3);
      const basePositions = origGeometry.attributes.position.array;
      
      for (let i = 0; i < origCount * 3; i++) {
        morphDelta[i] = morphTargetPositions[i] - basePositions[i];
      }
      
      // Set up morph targets on the geometry
      origGeometry.morphAttributes.position = [
        new THREE.BufferAttribute(morphDelta, 3)
      ];
      origGeometry.morphTargetsRelative = true;
      
      // Compute normals
      origGeometry.computeVertexNormals();
      
      setMorphGeometry(origGeometry);
      setIsReady(true);
      
      console.log('✅ MorphArtery: Morph targets ready!');
      
    } catch (err) {
      console.error('MorphArtery: Error setting up morph targets:', err);
    }
    
  }, [clonedOriginal, clonedHealed, visible]);
  
  /**
   * Update morph target influence when slider changes
   */
  useEffect(() => {
    if (meshRef.current && isReady) {
      meshRef.current.morphTargetInfluences[0] = morphValue;
    }
  }, [morphValue, isReady]);
  
  // Don't render until ready
  if (!morphGeometry || !visible || !isReady) return null;
  
  // Compute color based on morph value (transition from red to green)
  const color = new THREE.Color();
  color.setHSL(
    morphValue * 0.33, // Hue: 0 (red) to 0.33 (green)
    0.7,               // Saturation
    0.5                // Lightness
  );
  
  return (
    <mesh 
      ref={meshRef} 
      geometry={morphGeometry}
      morphTargetInfluences={[morphValue]}
    >
      <meshStandardMaterial
        color={color}
        side={THREE.DoubleSide}
        metalness={0.2}
        roughness={0.5}
        clippingPlanes={clippingPlanes}
        clipShadows={true}
      />
    </mesh>
  );
}

/**
 * MorphSlider - Bottom control bar for morph value
 */
export function MorphSlider({ morphValue = 0, onMorphChange, visible = true }) {
  if (!visible) return null;
  
  return (
    <div className="flex items-center gap-4 bg-slate-900/95 backdrop-blur-md rounded-xl px-6 py-4 border border-slate-700/50 shadow-2xl">
      {/* Left Label */}
      <div className="text-right min-w-[100px]">
        <p className="text-[9px] text-slate-500 uppercase tracking-[0.15em]">BASELINE</p>
        <p className="text-xs text-red-400 font-medium">Anatomy</p>
      </div>
      
      {/* Slider */}
      <div className="relative w-56">
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={morphValue}
          onChange={(e) => onMorphChange?.(parseFloat(e.target.value))}
          className="w-full h-2 rounded-full appearance-none cursor-pointer
                     [&::-webkit-slider-thumb]:appearance-none
                     [&::-webkit-slider-thumb]:w-5
                     [&::-webkit-slider-thumb]:h-5
                     [&::-webkit-slider-thumb]:bg-white
                     [&::-webkit-slider-thumb]:rounded-full
                     [&::-webkit-slider-thumb]:shadow-lg
                     [&::-webkit-slider-thumb]:cursor-pointer
                     [&::-webkit-slider-thumb]:border-2
                     [&::-webkit-slider-thumb]:border-slate-300
                     [&::-webkit-slider-thumb]:transition-transform
                     [&::-webkit-slider-thumb]:hover:scale-110
                     [&::-moz-range-thumb]:w-5
                     [&::-moz-range-thumb]:h-5
                     [&::-moz-range-thumb]:bg-white
                     [&::-moz-range-thumb]:rounded-full
                     [&::-moz-range-thumb]:border-0"
          style={{
            background: `linear-gradient(to right, 
              #ef4444 0%, 
              #f97316 25%,
              #eab308 50%, 
              #84cc16 75%,
              #22c55e 100%)`
          }}
        />
        
        {/* Percentage indicator */}
        <div className="absolute -bottom-6 left-0 right-0 flex justify-center">
          <span className={`text-xs font-mono px-2 py-0.5 rounded ${
            morphValue < 0.3 ? 'bg-red-500/20 text-red-400' :
            morphValue < 0.7 ? 'bg-yellow-500/20 text-yellow-400' :
            'bg-green-500/20 text-green-400'
          }`}>
            {Math.round(morphValue * 100)}%
          </span>
        </div>
      </div>
      
      {/* Right Label */}
      <div className="text-left min-w-[100px]">
        <p className="text-[9px] text-slate-500 uppercase tracking-[0.15em]">OPTIMIZED</p>
        <p className="text-xs text-emerald-400 font-medium">Geometry</p>
      </div>
    </div>
  );
}
