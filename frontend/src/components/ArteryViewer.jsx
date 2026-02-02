import { useRef, useEffect, useState, useMemo, Suspense, useCallback } from 'react';
import { Canvas, useLoader, useThree, useFrame } from '@react-three/fiber';
import {
  OrbitControls,
  OrthographicCamera,
  Environment,
  GizmoHelper,
  GizmoViewcube,
  Grid,
  ContactShadows,
  Line,
  Html
} from '@react-three/drei';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import * as THREE from 'three';
import DifferenceMesh, { HeatmapLegend } from './DifferenceMesh';
import MorphingMesh, { MorphSliderLegend } from './MorphingMesh';
import MorphArtery from './MorphArtery';

// ============================================
// CLIPPING PLANE HELPER
// ============================================
function ClippingPlaneHelper({ clippingY, visible }) {
  if (!visible || clippingY >= 2) return null;

  return (
    <mesh position={[0, clippingY, 0]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[4, 4]} />
      <meshBasicMaterial
        color="#06b6d4"
        transparent
        opacity={0.15}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  );
}

// ============================================
// MEASUREMENT TOOL - Point markers and line
// ============================================
function MeasurementTool({ points, onClearMeasurement }) {
  if (points.length === 0) return null;

  const midpoint = points.length === 2
    ? new THREE.Vector3().addVectors(points[0], points[1]).multiplyScalar(0.5)
    : null;

  const distance = points.length === 2
    ? points[0].distanceTo(points[1]) * 10 // Scale to mm (assuming model units)
    : null;

  return (
    <group>
      {/* Measurement points */}
      {points.map((point, i) => (
        <mesh key={i} position={point}>
          <sphereGeometry args={[0.03, 16, 16]} />
          <meshBasicMaterial color="#fbbf24" />
        </mesh>
      ))}

      {/* Line between points */}
      {points.length === 2 && (
        <>
          <Line
            points={[points[0], points[1]]}
            color="#fbbf24"
            lineWidth={2}
            dashed={true}
            dashScale={50}
            dashSize={0.05}
            gapSize={0.03}
          />

          {/* Distance label */}
          <Html position={midpoint} center>
            <div className="bg-amber-500 text-slate-900 px-2 py-1 rounded text-xs font-bold whitespace-nowrap shadow-lg flex items-center gap-2">
              <span>📏 {distance.toFixed(2)} mm</span>
              <button
                onClick={onClearMeasurement}
                className="hover:bg-amber-600 rounded px-1"
                title="Clear measurement"
              >
                ✕
              </button>
            </div>
          </Html>
        </>
      )}
    </group>
  );
}

// ============================================
// CLICKABLE MODEL - Handles raycasting for measurements
// ============================================
function ClickableModel({
  url,
  color,
  opacity = 1.0,
  wireframe = false,
  visible = true,
  clippingPlanes = [],
  isMeasuring = false,
  onMeasureClick
}) {
  const obj = useLoader(OBJLoader, url);
  const meshRef = useRef();
  const { raycaster, camera, pointer, gl } = useThree();

  // Clone the object to avoid mutations affecting shared instances
  const clonedObj = useMemo(() => obj.clone(true), [obj]);

  // Store mesh references for raycasting
  const meshesRef = useRef([]);

  useEffect(() => {
    if (clonedObj) {
      // Center the model
      const box = new THREE.Box3().setFromObject(clonedObj);
      const center = box.getCenter(new THREE.Vector3());
      clonedObj.position.sub(center);

      // Scale to fit
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const scale = 2 / maxDim;
      clonedObj.scale.setScalar(scale);

      meshesRef.current = [];

      // Apply material with clipping
      clonedObj.traverse((child) => {
        if (child.isMesh) {
          child.material = new THREE.MeshStandardMaterial({
            color: color,
            opacity: opacity,
            transparent: opacity < 1.0,
            wireframe: wireframe,
            metalness: wireframe ? 0.1 : 0.3,
            roughness: wireframe ? 0.8 : 0.4,
            side: THREE.DoubleSide,
            depthWrite: opacity >= 1.0,
            clippingPlanes: clippingPlanes,
            clipShadows: true,
          });
          child.castShadow = !wireframe;
          child.receiveShadow = !wireframe;
          child.visible = visible;
          meshesRef.current.push(child);
        }
      });
    }
  }, [clonedObj, color, opacity, wireframe, visible, clippingPlanes]);

  // Handle click for measurement
  const handleClick = useCallback((event) => {
    if (!isMeasuring || !onMeasureClick) return;

    event.stopPropagation();

    raycaster.setFromCamera(pointer, camera);
    const intersects = raycaster.intersectObjects(meshesRef.current, true);

    if (intersects.length > 0) {
      onMeasureClick(intersects[0].point.clone());
    }
  }, [isMeasuring, onMeasureClick, raycaster, pointer, camera]);

  useEffect(() => {
    if (isMeasuring) {
      gl.domElement.style.cursor = 'crosshair';
    } else {
      gl.domElement.style.cursor = 'grab';
    }
    return () => {
      gl.domElement.style.cursor = 'grab';
    };
  }, [isMeasuring, gl]);

  return clonedObj && visible ? (
    <primitive
      ref={meshRef}
      object={clonedObj}
      onClick={handleClick}
    />
  ) : null;
}

// Legacy OBJModel for backward compatibility
function OBJModel({ url, color, opacity = 1.0, wireframe = false, visible = true, clippingPlanes = [] }) {
  return (
    <ClickableModel
      url={url}
      color={color}
      opacity={opacity}
      wireframe={wireframe}
      visible={visible}
      clippingPlanes={clippingPlanes}
      isMeasuring={false}
    />
  );
}

// Main viewer component
export default function ArteryViewer({
  originalObjUrl,
  healedObjUrl,
  showComparison,
  wireframeMode = false,
  onWireframeModeChange,
  // New professional tool props
  clippingY = 2,           // Default: no clipping (above model)
  isMeasuring = false,
  onMeasurementUpdate,     // Callback to report measurements to parent
  // Morphing slider props (THESIS: Manifold Learning)
  morphValue = 0,          // 0 = original, 1 = healed
  onMorphValueChange,      // Callback to update morph value
  // Root cause heatmap props (THESIS: Root Cause Analysis)
  showHeatmap = false,     // Whether to show vertex colors on morph mesh
  onHeatmapChange,         // Callback to toggle heatmap
}) {
  const [error, setError] = useState(null);
  const [localWireframe, setLocalWireframe] = useState(false);
  // viewMode: 'anatomical' | 'heatmap' | 'morph'
  const [viewMode, setViewMode] = useState('anatomical');
  const [measurementPoints, setMeasurementPoints] = useState([]);

  // Use prop wireframe if provided, otherwise use local state
  const isWireframe = wireframeMode !== undefined ? wireframeMode : localWireframe;
  const setWireframe = onWireframeModeChange || setLocalWireframe;

  // Create clipping planes
  const clippingPlanes = useMemo(() => {
    if (clippingY >= 2) return []; // No clipping
    return [new THREE.Plane(new THREE.Vector3(0, -1, 0), clippingY)];
  }, [clippingY]);

  // Handle measurement point clicks
  const handleMeasureClick = useCallback((point) => {
    setMeasurementPoints(prev => {
      if (prev.length >= 2) {
        // Start new measurement
        return [point];
      }
      return [...prev, point];
    });
  }, []);

  // Update parent when measurement points change
  useEffect(() => {
    if (measurementPoints.length === 2) {
      const distance = measurementPoints[0].distanceTo(measurementPoints[1]) * 10; // Scale to mm
      onMeasurementUpdate?.(distance);
    } else if (measurementPoints.length < 2) {
      onMeasurementUpdate?.(null);
    }
  }, [measurementPoints, onMeasurementUpdate]);

  // Clear measurement
  const handleClearMeasurement = useCallback(() => {
    setMeasurementPoints([]);
  }, []);

  // Handle loading errors
  const handleError = (err) => {
    console.error('Error loading 3D model:', err);
    setError('Failed to load 3D model. Please check the file format.');
  };

  if (error) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-900 rounded-lg border border-red-500/20">
        <div className="text-center p-8">
          <svg className="w-16 h-16 mx-auto mb-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <p className="text-red-400 font-medium">{error}</p>
        </div>
      </div>
    );
  }

  if (!originalObjUrl && !healedObjUrl) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-900 rounded-lg border border-slate-700">
        <div className="text-center p-8">
          <svg className="w-20 h-20 mx-auto mb-6 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
          </svg>
          <p className="text-slate-400 text-lg font-medium mb-2">No 3D Model Loaded</p>
          <p className="text-slate-500 text-sm">Upload an artery scan to visualize</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-screen h-screen relative bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 overflow-hidden">
      <Canvas
        shadows
        gl={{
          antialias: true,
          alpha: true,
          localClippingEnabled: true // Enable clipping planes
        }}
        onError={handleError}
      >
        {/* Orthographic Camera - No perspective distortion for medical precision */}
        <OrthographicCamera
          makeDefault
          zoom={100}
          position={[3, 2, 3]}
          near={0.1}
          far={1000}
        />

        {/* Lighting */}
        <ambientLight intensity={0.5} />
        <directionalLight
          position={[10, 10, 5]}
          intensity={1.2}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
        />
        <directionalLight position={[-5, 5, -5]} intensity={0.4} />
        <pointLight position={[-10, -10, -5]} intensity={0.3} color="#4f46e5" />
        <pointLight position={[0, 5, 0]} intensity={0.2} color="#06b6d4" />

        {/* Environment */}
        <Environment preset="city" />

        {/* Professional Grid - Infinite faded grid for scale reference */}
        <Grid
          position={[0, -1.5, 0]}
          args={[20, 20]}
          cellSize={0.5}
          cellThickness={0.5}
          cellColor="#334155"
          sectionSize={2}
          sectionThickness={1}
          sectionColor="#475569"
          fadeDistance={15}
          fadeStrength={1}
          followCamera={false}
          infiniteGrid={true}
        />

        {/* Contact Shadows - Grounding effect */}
        <ContactShadows
          position={[0, -1.49, 0]}
          opacity={0.4}
          scale={10}
          blur={2}
          far={3}
          color="#000000"
        />

        {/* Clipping Plane Visual Helper */}
        <ClippingPlaneHelper clippingY={clippingY} visible={clippingY < 2} />

        {/*
          TRUE MEDICAL OVERLAY MODE:
          When showComparison is true:
          - Original: Solid Red (or Wireframe Red if wireframeMode is on)
          - Healed: Transparent Green - rendered INSIDE the original
          Both meshes overlap perfectly to show the aneurysm bulging out
          
          SCIENTIFIC HEATMAP MODE:
          When viewMode === 'heatmap':
          - Shows a single mesh with vertex colors (blue->red gradient)
          - Blue = no change (healthy tissue)
          - Red = high change (aneurysm site that shrank)
          
          MANIFOLD MORPH MODE:
          When viewMode === 'morph':
          - Interpolates between original and healed vertex positions
          - Uses morphValue (0.0 = original, 1.0 = healed)
          - Shows smooth transition along the learned manifold
        */}

        {/* Manifold Morph View - Uses proper Three.js morph targets (single mesh) */}
        {viewMode === 'morph' && originalObjUrl && healedObjUrl && showComparison && (
          <Suspense fallback={null}>
            <MorphArtery
              originalUrl={originalObjUrl}
              healedUrl={healedObjUrl}
              morphValue={morphValue}
              clippingPlanes={clippingPlanes}
              visible={true}
            />
          </Suspense>
        )}

        {/* Scientific Heatmap View */}
        {viewMode === 'heatmap' && originalObjUrl && healedObjUrl && showComparison && (
          <Suspense fallback={null}>
            <DifferenceMesh
              originalUrl={originalObjUrl}
              healedUrl={healedObjUrl}
              maxDistance={0.15}
              visible={true}
            />
          </Suspense>
        )}

        {/* Anatomical View (only show when NOT in heatmap or morph mode) */}
        {viewMode === 'anatomical' && (
          <>
            {/* Healed Model (Green, Transparent) - Render FIRST (inner healthy shape) */}
            {healedObjUrl && showComparison && (
              <ClickableModel
                url={healedObjUrl}
                color="#10b981"
                opacity={0.5}
                wireframe={false}
                visible={true}
                clippingPlanes={clippingPlanes}
                isMeasuring={isMeasuring}
                onMeasureClick={handleMeasureClick}
              />
            )}

            {/* Original Model - Render SECOND (outer diseased shape) */}
            {originalObjUrl && (
              <ClickableModel
                url={originalObjUrl}
                color={showComparison ? "#ef4444" : "#06b6d4"}
                opacity={showComparison ? (isWireframe ? 1.0 : 0.7) : 1.0}
                wireframe={showComparison && isWireframe}
                visible={true}
                clippingPlanes={clippingPlanes}
                isMeasuring={isMeasuring}
                onMeasureClick={handleMeasureClick}
              />
            )}
          </>
        )}

        {/* Measurement Tool */}
        <MeasurementTool
          points={measurementPoints}
          onClearMeasurement={handleClearMeasurement}
        />

        {/* Controls - Fluid physics-based camera */}
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          autoRotate={false}
          autoRotateSpeed={0.5}
          minZoom={20}
          maxZoom={500}
          // Smooth inertia for fluid camera feel
          enableDamping={true}
          dampingFactor={0.05}
          // Zoom to cursor position (not center)
          zoomToCursor={true}
          // Prevent getting lost in space
          minDistance={2}
          maxDistance={15}
        />

        {/* GizmoHelper - Medical view orientation cube */}
        <GizmoHelper
          alignment="bottom-right"
          margin={[100, 140]}
        >
          <GizmoViewcube
            color="#1e293b"
            textColor="#94a3b8"
            strokeColor="#475569"
            hoverColor="#0ea5e9"
            opacity={0.9}
            faces={['Right', 'Left', 'Top', 'Bottom', 'Front', 'Back']}
          />
        </GizmoHelper>
      </Canvas>

      {/* Legend - Only show in comparison mode */}
      {showComparison && (
        <div className="absolute bottom-28 left-8 bg-[rgba(20,20,20,0.8)] backdrop-blur-xl rounded-2xl p-4 border border-white/10 shadow-xl">
          {viewMode === 'heatmap' ? (
            <HeatmapLegend />
          ) : viewMode === 'morph' ? (
            <div className="flex flex-col gap-3">
              <MorphSliderLegend morphValue={morphValue} onMorphChange={onMorphValueChange} />
              {/* Root Cause Toggle in Legend */}
              {showHeatmap && (
                <div className="border-t border-slate-700 pt-2">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-purple-400 text-xs font-medium">🎯 Root Cause Active</span>
                  </div>
                  <div className="w-full h-2 rounded"
                    style={{
                      background: 'linear-gradient(to right, #3b82f6, #06b6d4, #10b981, #eab308, #ef4444)'
                    }}
                  />
                  <div className="flex justify-between text-xs text-slate-500 mt-1">
                    <span>Healthy</span>
                    <span>Aneurysm</span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col gap-2 text-sm">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <div className={`w-4 h-4 rounded ${isWireframe ? 'border-2 border-red-500 bg-transparent' : 'bg-red-500'}`}></div>
                  <span className="text-slate-300">Original (Aneurysm)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded bg-green-500/50 border border-green-500"></div>
                  <span className="text-slate-300">Healed (Healthy)</span>
                </div>
              </div>
              <p className="text-xs text-slate-500 italic">
                Red bulges beyond green = aneurysm location
              </p>
            </div>
          )}
        </div>
      )}

      {/* View Mode Toggle - Top Center (below status indicators) */}
      {showComparison && healedObjUrl && (
        <div className="absolute top-28 left-1/2 -translate-x-1/2 bg-[rgba(15,23,42,0.8)] backdrop-blur-xl rounded-2xl p-2 border border-slate-700/50 flex gap-2 shadow-xl z-40">
          <button
            onClick={() => setViewMode('anatomical')}
            className={`px-3 py-2 rounded-xl text-xs font-medium transition-all flex items-center gap-1 ${viewMode === 'anatomical'
              ? 'bg-cyan-500/30 text-cyan-300 ring-1 ring-cyan-500/50'
              : 'bg-white/5 text-slate-400 hover:bg-white/10'
              }`}
            title="Anatomical View: Red/Green overlay"
          >
            🔴 Anatomical
          </button>
          <button
            onClick={() => setViewMode('heatmap')}
            className={`px-3 py-2 rounded-xl text-xs font-medium transition-all flex items-center gap-1 ${viewMode === 'heatmap'
              ? 'bg-purple-500/30 text-purple-300 ring-1 ring-purple-500/50'
              : 'bg-white/5 text-slate-400 hover:bg-white/10'
              }`}
            title="Scientific View: Blue-to-Red heatmap"
          >
            🔬 Heatmap
          </button>
          <button
            onClick={() => setViewMode('morph')}
            className={`px-3 py-2 rounded-xl text-xs font-medium transition-all flex items-center gap-1 ${viewMode === 'morph'
              ? 'bg-green-500/30 text-green-300 ring-1 ring-green-500/50'
              : 'bg-white/5 text-slate-400 hover:bg-white/10'
              }`}
            title="Manifold View: Smooth morphing between states"
          >
            🔄 Morph
          </button>
          {viewMode === 'anatomical' && (
            <button
              onClick={() => setWireframe(!isWireframe)}
              className={`px-3 py-2 rounded-xl text-xs font-medium transition-all ${isWireframe
                ? 'bg-cyan-500/30 text-cyan-300 ring-1 ring-cyan-500/50'
                : 'bg-white/5 text-slate-400 hover:bg-white/10'
                }`}
            >
              {isWireframe ? '🔲 Wire' : '🔳 Solid'}
            </button>
          )}
        </div>
      )}


    </div>
  );
}
