import { Suspense, useMemo, useRef, useState } from 'react';
import { Canvas, useLoader } from '@react-three/fiber';
import {
  OrbitControls,
  OrthographicCamera,
  GizmoHelper,
  GizmoViewcube,
  Grid,
  ContactShadows,
} from '@react-three/drei';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import * as THREE from 'three';

// ============================================
// Compute mm-preserving transform
// ----------------------------------------------
// We translate the mesh to the origin (subtract centroid) and uniformly
// scale it so the longest bbox edge fits inside a ~3-unit display box.
// We record centroid (in mm) and scale, so we can invert the mapping
// for any point the user clicks in display space.
// ============================================
function computeVesselTransforms(object3d) {
  const box = new THREE.Box3().setFromObject(object3d);
  const centroid = new THREE.Vector3();
  box.getCenter(centroid);
  const size = new THREE.Vector3();
  box.getSize(size);
  const maxExtent = Math.max(size.x, size.y, size.z) || 1;
  const targetExtent = 3; // display box edge length
  const scale = targetExtent / maxExtent;
  return {
    centroidMm: centroid.clone(), // centroid in the ORIGINAL (mm) space
    scale,
    bboxMmMin: box.min.clone(),
    bboxMmMax: box.max.clone(),
  };
}

// displayToMm: inverse of the centroid+scale mapping.
// Display transform applied to the mesh group is:
//   display = (mm - centroidMm) * scale
// So the inverse is:
//   mm = display / scale + centroidMm
function makeDisplayToMm(transforms) {
  return (displayVec3) => {
    const mm = new THREE.Vector3(
      displayVec3.x / transforms.scale + transforms.centroidMm.x,
      displayVec3.y / transforms.scale + transforms.centroidMm.y,
      displayVec3.z / transforms.scale + transforms.centroidMm.z,
    );
    return [mm.x, mm.y, mm.z];
  };
}

function VesselMesh({ url, onReady, onClickDisplay }) {
  const obj = useLoader(OBJLoader, url);
  const groupRef = useRef();
  const [transforms] = useState(() => computeVesselTransforms(obj));

  useMemo(() => {
    obj.traverse((child) => {
      if (child.isMesh) {
        child.material = new THREE.MeshStandardMaterial({
          color: '#67B8D6',
          roughness: 0.4,
          metalness: 0.1,
        });
      }
    });
  }, [obj]);

  // Report transforms up once
  useMemo(() => { onReady?.(transforms); }, [transforms, onReady]);

  return (
    <group
      ref={groupRef}
      scale={[transforms.scale, transforms.scale, transforms.scale]}
      position={[
        -transforms.centroidMm.x * transforms.scale,
        -transforms.centroidMm.y * transforms.scale,
        -transforms.centroidMm.z * transforms.scale,
      ]}
      onClick={(e) => {
        e.stopPropagation();
        // event.point is in world (display) coordinates
        onClickDisplay(e.point.clone());
      }}
    >
      <primitive object={obj} />
    </group>
  );
}

export default function VesselTreeViewer({ apiUrl, sessionId, onCropPointSelected }) {
  const url = `${apiUrl}/dicom/full-mesh/${sessionId}.obj`;

  const [transforms, setTransforms] = useState(null);
  const [clickDisplay, setClickDisplay] = useState(null); // THREE.Vector3 in display coords
  const [clickMm, setClickMm] = useState(null); // [x,y,z] in mm
  const [cropRadius, setCropRadius] = useState(15); // mm

  const displayToMm = useMemo(
    () => (transforms ? makeDisplayToMm(transforms) : null),
    [transforms],
  );

  const handleClick = (displayVec) => {
    if (!displayToMm) return;
    const mm = displayToMm(displayVec);
    setClickDisplay(displayVec);
    setClickMm(mm);
  };

  const handleAnalyze = () => {
    if (!clickMm) return;
    onCropPointSelected(clickMm, cropRadius);
  };

  const displayRadius = transforms ? cropRadius * transforms.scale : 0;

  const bboxStr = transforms
    ? `[${transforms.bboxMmMin.x.toFixed(1)}, ${transforms.bboxMmMin.y.toFixed(1)}, ${transforms.bboxMmMin.z.toFixed(1)}] to [${transforms.bboxMmMax.x.toFixed(1)}, ${transforms.bboxMmMax.y.toFixed(1)}, ${transforms.bboxMmMax.z.toFixed(1)}]`
    : '—';

  return (
    <div className="relative w-full h-full">
      <Canvas
        shadows
        style={{ width: '100%', height: '100%' }}
        gl={{ antialias: true, alpha: true, preserveDrawingBuffer: true }}
      >
        <OrthographicCamera makeDefault zoom={100} position={[3, 2, 3]} near={0.1} far={1000} />

        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 10, 5]} intensity={0.9} castShadow />
        <directionalLight position={[-5, 5, -5]} intensity={0.3} />
        <pointLight position={[0, 5, 0]} intensity={0.15} color="#4A9EFF" />

        <Grid
          position={[0, -1.5, 0]}
          args={[20, 20]}
          cellSize={0.5}
          cellThickness={0.3}
          cellColor="#1a1e28"
          sectionSize={2}
          sectionThickness={0.5}
          sectionColor="#242836"
          fadeDistance={15}
          fadeStrength={1.5}
          followCamera={false}
          infiniteGrid={true}
        />
        <ContactShadows position={[0, -1.49, 0]} opacity={0.3} scale={10} blur={2.5} far={3} color="#000000" />

        <Suspense fallback={null}>
          <VesselMesh url={url} onReady={setTransforms} onClickDisplay={handleClick} />
        </Suspense>

        {/* Click marker */}
        {clickDisplay && (
          <>
            <mesh position={[clickDisplay.x, clickDisplay.y, clickDisplay.z]}>
              <sphereGeometry args={[0.05, 24, 24]} />
              <meshStandardMaterial color="#ef4444" emissive="#ef4444" emissiveIntensity={0.4} />
            </mesh>
            <mesh position={[clickDisplay.x, clickDisplay.y, clickDisplay.z]}>
              <sphereGeometry args={[displayRadius, 32, 32]} />
              <meshStandardMaterial color="#ef4444" transparent opacity={0.15} depthWrite={false} />
            </mesh>
          </>
        )}

        <OrbitControls
          makeDefault
          enablePan
          enableZoom
          enableRotate
          minZoom={20}
          maxZoom={500}
          enableDamping
          dampingFactor={0.05}
          zoomToCursor
          minDistance={2}
          maxDistance={15}
        />

        <GizmoHelper alignment="bottom-right" margin={[100, 140]}>
          <GizmoViewcube
            color="#1A1D27"
            textColor="#94A3B8"
            strokeColor="#2a2e3a"
            hoverColor="#4A9EFF"
            opacity={0.9}
            faces={['Right', 'Left', 'Top', 'Bottom', 'Front', 'Back']}
          />
        </GizmoHelper>
      </Canvas>

      {/* DEBUG OVERLAY — top-left */}
      <div
        className="absolute top-3 left-3"
        style={{
          background: 'rgba(15, 17, 23, 0.85)',
          backdropFilter: 'blur(12px)',
          border: '1px solid rgba(255,255,255,0.06)',
          borderRadius: 8,
          padding: '8px 10px',
          fontFamily: 'ui-monospace, SFMono-Regular, monospace',
          fontSize: 10,
          color: '#94A3B8',
          lineHeight: 1.55,
          pointerEvents: 'none',
          maxWidth: 360,
        }}
      >
        <div>
          Click point (display):{' '}
          {clickDisplay
            ? `[${clickDisplay.x.toFixed(3)}, ${clickDisplay.y.toFixed(3)}, ${clickDisplay.z.toFixed(3)}]`
            : '—'}
        </div>
        <div>
          Click point (mm):{' '}
          {clickMm ? `[${clickMm[0].toFixed(2)}, ${clickMm[1].toFixed(2)}, ${clickMm[2].toFixed(2)}]` : '—'}
        </div>
        <div>Mesh bbox (mm): {bboxStr}</div>
      </div>

      {/* Crop controls — bottom center */}
      <div
        className="absolute bottom-5 left-1/2 -translate-x-1/2 flex items-center gap-3"
        style={{
          background: 'rgba(26, 29, 39, 0.9)',
          backdropFilter: 'blur(24px)',
          border: '1px solid rgba(255,255,255,0.06)',
          borderRadius: 20,
          padding: '8px 16px',
        }}
      >
        <span style={{ fontSize: 11, color: '#64748B', fontWeight: 400 }}>Crop radius</span>
        <input
          type="range"
          min={5}
          max={40}
          step={1}
          value={cropRadius}
          onChange={(e) => setCropRadius(parseInt(e.target.value, 10))}
          style={{ width: 140, accentColor: '#4A9EFF' }}
        />
        <span
          style={{
            fontSize: 11,
            color: '#F1F5F9',
            fontFamily: 'ui-monospace, monospace',
            minWidth: 40,
          }}
        >
          {cropRadius} mm
        </span>
        <button
          onClick={handleAnalyze}
          disabled={!clickMm}
          style={{
            padding: '6px 16px',
            background: clickMm ? 'linear-gradient(135deg, #4A9EFF, #3B82F6)' : 'rgba(255,255,255,0.04)',
            color: clickMm ? '#fff' : '#475569',
            fontWeight: 500,
            fontSize: 12,
            borderRadius: 14,
            border: 'none',
            cursor: clickMm ? 'pointer' : 'not-allowed',
            boxShadow: clickMm ? '0 4px 16px rgba(74, 158, 255, 0.25)' : 'none',
          }}
        >
          Analyze this region
        </button>
      </div>
    </div>
  );
}

