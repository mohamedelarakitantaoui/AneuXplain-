import { useRef, useEffect, useState, useMemo, Suspense, useCallback, forwardRef, useImperativeHandle } from 'react';
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

// ============================================
// SHARED MESH TRANSFORMS
// ============================================

/**
 * Compute centering offset and uniform scale for an OBJ so it fits in a
 * normalized 2-unit bounding box centered at the origin.
 * Returns { center: Vector3, scale: number }
 */
function computeMeshTransforms(obj) {
  const box = new THREE.Box3().setFromObject(obj);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const scale = 2 / maxDim;

  console.log('[computeMeshTransforms] Raw OBJ bounds:', {
    min: box.min.toArray().map(v => v.toFixed(2)),
    max: box.max.toArray().map(v => v.toFixed(2)),
    center: center.toArray().map(v => v.toFixed(2)),
    size: size.toArray().map(v => v.toFixed(2)),
    maxDim: maxDim.toFixed(2),
    scale: scale.toFixed(4),
  });

  return { center, scale };
}

/**
 * Apply shared centering + scale to a cloned OBJ group.
 * Must use the SAME transforms for ClickableModel and HeatmapMesh
 * so they align perfectly and the camera never jumps.
 */
function applyMeshTransforms(obj, transforms) {
  // Three.js applies: worldPos = (localPos * scale) + position
  // To centre a vertex at `center` to origin: position = -center * scale
  const s = transforms.scale;
  obj.scale.setScalar(s);
  obj.position.set(
    -transforms.center.x * s,
    -transforms.center.y * s,
    -transforms.center.z * s,
  );
}


// ============================================
// CAMERA CONTROLLER — auto-frame on load, reset on demand
// ============================================

/**
 * Sits inside the Canvas. On first mount (or when resetKey changes),
 * frames the mesh so it fills ~70% of the orthographic viewport.
 *
 * Uses useFrame (not useEffect) so the framing runs INSIDE the render
 * loop — after OrbitControls has initialised — preventing the controls
 * from overriding the camera on their next update().
 */
function CameraController({ transforms, resetKey, controlsRef }) {
  const { camera, gl } = useThree();
  const framedKeyRef = useRef(null);
  const pendingFrameRef = useRef(false);

  // Flag that we need to frame — actual work happens in useFrame
  useEffect(() => {
    if (!transforms || framedKeyRef.current === resetKey) return;
    framedKeyRef.current = resetKey;
    pendingFrameRef.current = true;
  }, [transforms, resetKey]);

  useFrame(() => {
    if (!pendingFrameRef.current) return;
    pendingFrameRef.current = false;

    // Mesh is scaled to fit in a 2-unit box centered at origin
    const meshCenter = new THREE.Vector3(0, 0, 0);

    // For orthographic camera: set zoom so the 2-unit box fills ~70% of viewport
    const aspect = gl.domElement.clientWidth / gl.domElement.clientHeight;
    const fitH = 2 / 0.7;
    const fitV = 2 / 0.7;
    const fitSize = Math.max(fitH, fitV / aspect);

    const viewHeight = gl.domElement.clientHeight;
    const newZoom = viewHeight / fitSize;

    camera.zoom = Math.max(newZoom, 50);
    camera.position.set(3, 2, 3);
    camera.lookAt(meshCenter);
    camera.updateProjectionMatrix();

    // Sync OrbitControls target so orbiting centres on the mesh
    const controls = controlsRef?.current;
    if (controls) {
      controls.target.copy(meshCenter);
      controls.update();
    }

    console.log('[CameraController] Auto-frame fired:', {
      meshCenter: meshCenter.toArray(),
      cameraPos: camera.position.toArray(),
      zoom: camera.zoom,
      viewportH: viewHeight,
      aspect: aspect.toFixed(2),
      controlsTarget: controls ? controls.target.toArray() : 'null',
      controlsReady: !!controls,
    });
  });

  return null;
}

// ============================================
// CLIPPING PLANE HELPER
// ============================================
function ClippingPlaneHelper({ clippingY, visible }) {
  if (!visible || clippingY >= 2) return null;

  return (
    <mesh position={[0, clippingY, 0]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[4, 4]} />
      <meshBasicMaterial
        color="#4A9EFF"
        transparent
        opacity={0.08}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  );
}

// ============================================
// MEASUREMENT TOOL - Clean white style
// ============================================
function MeasurementTool({ points, onClearMeasurement }) {
  if (points.length === 0) return null;

  const midpoint = points.length === 2
    ? new THREE.Vector3().addVectors(points[0], points[1]).multiplyScalar(0.5)
    : null;

  const distance = points.length === 2
    ? points[0].distanceTo(points[1]) * 10
    : null;

  return (
    <group>
      {points.map((point, i) => (
        <group key={i}>
          <mesh position={point}>
            <sphereGeometry args={[0.025, 16, 16]} />
            <meshBasicMaterial color="#ffffff" />
          </mesh>
          <mesh position={point}>
            <sphereGeometry args={[0.04, 16, 16]} />
            <meshBasicMaterial color="#4A9EFF" transparent opacity={0.2} />
          </mesh>
        </group>
      ))}

      {points.length === 2 && (
        <>
          <Line
            points={[points[0], points[1]]}
            color="#ffffff"
            lineWidth={1.5}
            dashed={true}
            dashScale={50}
            dashSize={0.05}
            gapSize={0.03}
          />

          <Html position={midpoint} center>
            <div style={{
              background: '#242836',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: 20,
              padding: '4px 12px',
              fontSize: 12,
              fontWeight: 300,
              color: '#F1F5F9',
              whiteSpace: 'nowrap',
              boxShadow: '0 4px 16px rgba(0,0,0,0.3)',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              fontFamily: 'Inter, system-ui, sans-serif',
              fontVariantNumeric: 'tabular-nums',
            }}>
              <span>{distance.toFixed(2)} mm</span>
              <button
                onClick={onClearMeasurement}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#64748B',
                  cursor: 'pointer',
                  fontSize: 11,
                  padding: '0 2px',
                  lineHeight: 1,
                }}
                title="Clear"
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
// MEASUREMENT HIGHLIGHT OVERLAYS
// ============================================
/**
 * Renders spatial overlays (spheres, lines, regions) for the active
 * morphological parameter. Uses vertex_positions from the backend
 * (in original mesh space) and transforms them to world space using
 * the same centering + scaling applied to the 3D mesh.
 */
function MeasurementHighlight({ highlight, transforms }) {
  if (!highlight || !transforms) return null;

  const { spatial } = highlight;
  if (!spatial || !spatial.vertex_positions || spatial.vertex_positions.length === 0) return null;

  console.log('[MeasurementHighlight] rendering:', highlight.parameterName, spatial.type,
    'positions:', spatial.vertex_positions.length);

  const color = spatial.color || '#4A9EFF';
  const { center, scale } = transforms;

  // Transform backend positions (original mesh coords) → world space
  const worldPos = spatial.vertex_positions.map(([x, y, z]) =>
    new THREE.Vector3(
      (x - center.x) * scale,
      (y - center.y) * scale,
      (z - center.z) * scale,
    )
  );

  const sphereRadius = 0.03; // mesh is normalized to ~2 world units

  if (spatial.type === 'line' && worldPos.length >= 2) {
    const a = worldPos[0];
    const b = worldPos[1];
    const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
    const dir = new THREE.Vector3().subVectors(b, a);
    const length = dir.length();
    dir.normalize();

    const up = new THREE.Vector3(0, 1, 0);
    const quat = new THREE.Quaternion().setFromUnitVectors(up, dir);

    return (
      <group>
        <mesh position={mid} quaternion={quat}>
          <cylinderGeometry args={[sphereRadius * 0.25, sphereRadius * 0.25, length, 8]} />
          <meshBasicMaterial color={color} transparent opacity={0.6} />
        </mesh>
        {[a, b].map((pt, i) => (
          <group key={i}>
            <mesh position={pt}>
              <sphereGeometry args={[sphereRadius * 0.6, 16, 16]} />
              <meshBasicMaterial color="#ffffff" />
            </mesh>
            <mesh position={pt}>
              <sphereGeometry args={[sphereRadius, 16, 16]} />
              <meshBasicMaterial color={color} transparent opacity={0.35} />
            </mesh>
          </group>
        ))}
      </group>
    );
  }

  if (spatial.type === 'point') {
    return (
      <group>
        {worldPos.map((v, i) => (
          <group key={i}>
            <mesh position={v}>
              <sphereGeometry args={[sphereRadius * 0.6, 16, 16]} />
              <meshBasicMaterial color="#ffffff" />
            </mesh>
            <mesh position={v}>
              <sphereGeometry args={[sphereRadius, 16, 16]} />
              <meshBasicMaterial color={color} transparent opacity={0.35} />
            </mesh>
          </group>
        ))}
      </group>
    );
  }

  if (spatial.type === 'region') {
    return (
      <group>
        {worldPos.map((v, i) => (
          <mesh key={i} position={v}>
            <sphereGeometry args={[sphereRadius * 0.6, 8, 8]} />
            <meshBasicMaterial color={color} transparent opacity={0.55} />
          </mesh>
        ))}
      </group>
    );
  }

  return null;
}


// ============================================
// CAMERA ZOOM TO HIGHLIGHTED REGION
// ============================================
function animateCameraTo(camera, controls, targetLookAt, targetZoom, duration = 500) {
  const start = Date.now();
  const startZoom = camera.zoom;
  const startTarget = controls.target.clone();

  function update() {
    const t = Math.min((Date.now() - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3); // ease-out cubic

    camera.zoom = startZoom + (targetZoom - startZoom) * ease;
    camera.updateProjectionMatrix();
    controls.target.lerpVectors(startTarget, targetLookAt, ease);
    controls.update();

    if (t < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

function HighlightZoomController({ activeHighlight, transforms, controlsRef }) {
  const { camera, gl } = useThree();
  const homeStateRef = useRef(null);
  const prevHighlightRef = useRef(null);

  useEffect(() => {
    const controls = controlsRef?.current;
    if (!controls || !transforms) return;

    const hasHighlight = activeHighlight?.spatial?.vertex_positions?.length > 0;
    const hadHighlight = prevHighlightRef.current != null;
    prevHighlightRef.current = hasHighlight ? activeHighlight : null;

    if (hasHighlight) {
      // Save home state before first zoom
      if (!homeStateRef.current) {
        homeStateRef.current = {
          zoom: camera.zoom,
          target: controls.target.clone(),
        };
      }

      const { center, scale } = transforms;
      const positions = activeHighlight.spatial.vertex_positions.map(([x, y, z]) =>
        new THREE.Vector3(
          (x - center.x) * scale,
          (y - center.y) * scale,
          (z - center.z) * scale,
        )
      );

      // Bounding box of the highlighted region
      const regionBox = new THREE.Box3();
      positions.forEach(p => regionBox.expandByPoint(p));
      const regionCenter = regionBox.getCenter(new THREE.Vector3());
      const regionSize = regionBox.getSize(new THREE.Vector3());
      const maxRegionDim = Math.max(regionSize.x, regionSize.y, regionSize.z, 0.05);

      // Zoom so region fills ~60% of viewport
      const viewHeight = gl.domElement.clientHeight;
      const desiredWorldHeight = maxRegionDim / 0.6;
      const targetZoom = Math.min(Math.max(viewHeight / desiredWorldHeight, 50), 400);

      console.log('[HighlightZoom] Zooming to region:', {
        regionCenter: regionCenter.toArray().map(v => v.toFixed(3)),
        maxRegionDim: maxRegionDim.toFixed(3),
        targetZoom: targetZoom.toFixed(0),
      });

      animateCameraTo(camera, controls, regionCenter, targetZoom, 500);
    } else if (hadHighlight && homeStateRef.current) {
      // Restore home state
      console.log('[HighlightZoom] Restoring home camera');
      animateCameraTo(camera, controls, homeStateRef.current.target, homeStateRef.current.zoom, 500);
      homeStateRef.current = null;
    }
  }, [activeHighlight]); // eslint-disable-line react-hooks/exhaustive-deps

  return null;
}


// ============================================
// GRADIENT HEATMAP MESH
// ============================================

function heatmapColor(t) {
  const c = new THREE.Color();
  const hue = (1 - t) * 240 / 360;
  c.setHSL(hue, 1.0, 0.5);
  return c;
}

function normalizeHeatmapData(rawData) {
  // Backend already returns globally-normalized, risk-scaled values in [0, 1].
  // No re-normalization needed — just clamp for safety.
  return rawData.map(v => Math.max(0, Math.min(1, v)));
}

/**
 * HeatmapMesh — always mounted, uses shared transforms, toggled via `visible`.
 */
function HeatmapMesh({ url, heatmapData, clippingPlanes = [], transforms, visible }) {
  const obj = useLoader(OBJLoader, url);
  const meshRef = useRef();
  const initializedRef = useRef(false);

  const clonedObj = useMemo(() => obj.clone(true), [obj]);

  const normalizedHeatmap = useMemo(() => {
    if (!heatmapData || heatmapData.length === 0) return null;
    return normalizeHeatmapData(heatmapData);
  }, [heatmapData]);

  // One-time: apply shared transforms
  useEffect(() => {
    if (!clonedObj || !transforms || initializedRef.current) return;
    initializedRef.current = true;
    applyMeshTransforms(clonedObj, transforms);
  }, [clonedObj, transforms]);

  // Apply heatmap colors when data arrives
  useEffect(() => {
    if (!clonedObj || !normalizedHeatmap || normalizedHeatmap.length === 0 || !initializedRef.current) return;

    clonedObj.traverse((child) => {
      if (!child.isMesh) return;

      const geo = child.geometry;
      const pos = geo.getAttribute('position');
      if (!pos) return;

      const vertexCount = pos.count;

      const verts = [];
      for (let i = 0; i < vertexCount; i++) {
        verts.push(new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i)));
      }

      const centroid = new THREE.Vector3();
      verts.forEach(v => centroid.add(v));
      centroid.divideScalar(vertexCount);

      let maxDist = 0;
      const normalizedVerts = verts.map(v => {
        const d = v.clone().sub(centroid);
        const dist = d.length();
        if (dist > maxDist) maxDist = dist;
        return d;
      });
      if (maxDist > 0) {
        normalizedVerts.forEach(v => v.divideScalar(maxDist));
      }

      const faces = geo.index
        ? Array.from({ length: geo.index.count / 3 }, (_, fi) => [
            geo.index.getX(fi * 3),
            geo.index.getX(fi * 3 + 1),
            geo.index.getX(fi * 3 + 2),
          ])
        : Array.from({ length: vertexCount / 3 }, (_, fi) => [fi * 3, fi * 3 + 1, fi * 3 + 2]);

      const faceAreas = [];
      let totalArea = 0;
      for (const [a, b, c] of faces) {
        const va = verts[a], vb = verts[b], vc = verts[c];
        if (!va || !vb || !vc) { faceAreas.push(0); continue; }
        const ab = new THREE.Vector3().subVectors(vb, va);
        const ac = new THREE.Vector3().subVectors(vc, va);
        const area = ab.cross(ac).length() * 0.5;
        faceAreas.push(area);
        totalArea += area;
      }

      const numSamples = normalizedHeatmap.length;
      const samplePositions = [];

      const cumAreas = [];
      let cumSum = 0;
      for (const a of faceAreas) {
        cumSum += a;
        cumAreas.push(cumSum);
      }

      let seed = 42;
      const pseudoRandom = () => {
        seed = (seed * 16807 + 0) % 2147483647;
        return seed / 2147483647;
      };

      for (let s = 0; s < numSamples; s++) {
        const r = pseudoRandom() * totalArea;
        let fi = cumAreas.findIndex(c => c >= r);
        if (fi < 0) fi = faces.length - 1;

        const [a, b, c] = faces[fi];
        const va = verts[a], vb = verts[b], vc = verts[c];
        if (!va || !vb || !vc) {
          samplePositions.push(new THREE.Vector3(0, 0, 0));
          continue;
        }

        let u = pseudoRandom(), v = pseudoRandom();
        if (u + v > 1) { u = 1 - u; v = 1 - v; }
        const pt = va.clone()
          .add(new THREE.Vector3().subVectors(vb, va).multiplyScalar(u))
          .add(new THREE.Vector3().subVectors(vc, va).multiplyScalar(v));
        samplePositions.push(pt);
      }

      const colors = new Float32Array(vertexCount * 3);

      for (let vi = 0; vi < vertexCount; vi++) {
        const vert = verts[vi];
        let minDist = Infinity;
        let bestIdx = 0;

        for (let si = 0; si < samplePositions.length; si++) {
          const dx = vert.x - samplePositions[si].x;
          const dy = vert.y - samplePositions[si].y;
          const dz = vert.z - samplePositions[si].z;
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < minDist) {
            minDist = d2;
            bestIdx = si;
          }
        }

        const val = normalizedHeatmap[bestIdx];
        const col = heatmapColor(val);
        colors[vi * 3] = col.r;
        colors[vi * 3 + 1] = col.g;
        colors[vi * 3 + 2] = col.b;
      }

      geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

      child.material = new THREE.MeshStandardMaterial({
        vertexColors: true,
        metalness: 0.1,
        roughness: 0.6,
        side: THREE.DoubleSide,
        clippingPlanes,
        clipShadows: true,
      });
    });
  }, [clonedObj, normalizedHeatmap, clippingPlanes]);

  // Toggle visibility without unmounting
  useEffect(() => {
    if (!clonedObj) return;
    clonedObj.traverse((child) => {
      if (child.isMesh) child.visible = visible;
    });
  }, [clonedObj, visible]);

  return <primitive ref={meshRef} object={clonedObj} />;
}

function HeatmapLegend() {
  return (
    <div style={{
      background: 'rgba(15, 17, 23, 0.85)',
      backdropFilter: 'blur(24px)',
      borderRadius: 10,
      padding: 12,
      border: '1px solid rgba(255,255,255,0.06)',
    }}>
      <p style={{ fontSize: 11, color: '#94A3B8', marginBottom: 6 }}>Risk Sensitivity</p>
      <div
        style={{
          width: 200,
          height: 10,
          borderRadius: 5,
          background: 'linear-gradient(to right, #0000ff 0%, #00ffff 25%, #00ff00 50%, #ffff00 75%, #ff0000 100%)',
        }}
      />
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
        <span style={{ fontSize: 10, color: '#64748B' }}>Low</span>
        <span style={{ fontSize: 10, color: '#64748B' }}>High</span>
      </div>
    </div>
  );
}


// ============================================
// CLICKABLE MODEL — always mounted, visibility toggled
// ============================================
function ClickableModel({
  url,
  color,
  opacity = 1.0,
  wireframe = false,
  visible = true,
  clippingPlanes = [],
  isMeasuring = false,
  onMeasureClick,
  objRef,
  transforms,
}) {
  const obj = useLoader(OBJLoader, url);
  const meshRef = useRef();
  const { raycaster, camera, pointer, gl } = useThree();

  const clonedObj = useMemo(() => obj.clone(true), [obj]);
  const meshesRef = useRef([]);
  const initializedRef = useRef(false);

  // One-time geometry setup using shared transforms
  useEffect(() => {
    if (!clonedObj || !transforms || initializedRef.current) return;
    initializedRef.current = true;

    applyMeshTransforms(clonedObj, transforms);

    // Verify the mesh is now centred at origin
    const postBox = new THREE.Box3().setFromObject(clonedObj);
    console.log('[ClickableModel] After transforms applied:', {
      position: clonedObj.position.toArray().map(v => v.toFixed(2)),
      scale: clonedObj.scale.toArray().map(v => v.toFixed(4)),
      postBoundsMin: postBox.min.toArray().map(v => v.toFixed(2)),
      postBoundsMax: postBox.max.toArray().map(v => v.toFixed(2)),
      postCenter: postBox.getCenter(new THREE.Vector3()).toArray().map(v => v.toFixed(2)),
    });

    meshesRef.current = [];

    clonedObj.traverse((child) => {
      if (child.isMesh) {
        child.material = new THREE.MeshStandardMaterial({
          color: color,
          opacity: opacity,
          transparent: opacity < 1.0,
          wireframe: wireframe,
          metalness: wireframe ? 0.05 : 0.1,
          roughness: wireframe ? 0.9 : 0.6,
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

    if (objRef) objRef.current = clonedObj;
  }, [clonedObj, transforms]); // eslint-disable-line react-hooks/exhaustive-deps

  // Material-only updates — never touches geometry or camera
  useEffect(() => {
    if (!initializedRef.current || meshesRef.current.length === 0) return;

    meshesRef.current.forEach((child) => {
      child.material.color.set(color);
      child.material.opacity = opacity;
      child.material.transparent = opacity < 1.0;
      child.material.wireframe = wireframe;
      child.material.metalness = wireframe ? 0.05 : 0.1;
      child.material.roughness = wireframe ? 0.9 : 0.6;
      child.material.depthWrite = opacity >= 1.0;
      child.material.clippingPlanes = clippingPlanes;
      child.material.needsUpdate = true;
      child.castShadow = !wireframe;
      child.receiveShadow = !wireframe;
      child.visible = visible;
    });
  }, [color, opacity, wireframe, visible, clippingPlanes]);

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

  return clonedObj ? (
    <primitive
      ref={meshRef}
      object={clonedObj}
      onClick={handleClick}
    />
  ) : null;
}


// ============================================
// MAIN VIEWER COMPONENT
// ============================================
const ArteryViewer = forwardRef(function ArteryViewer({
  originalObjUrl,
  wireframeMode = false,
  onWireframeModeChange,
  clippingY = 2,
  isMeasuring = false,
  onMeasurementUpdate,
  activeHighlight = null,
  heatmapData = null,
  viewMode = 'risk',
  resetViewKey = 0,
}, ref) {
  const [error, setError] = useState(null);
  const [localWireframe, setLocalWireframe] = useState(false);
  const [measurementPoints, setMeasurementPoints] = useState([]);
  const canvasContainerRef = useRef(null);

  // Expose captureCanvas method to parent via ref
  useImperativeHandle(ref, () => ({
    captureCanvas: () => {
      const container = canvasContainerRef.current;
      if (!container) return null;
      const canvas = container.querySelector('canvas');
      if (!canvas) return null;
      try {
        return canvas.toDataURL('image/png');
      } catch {
        return null;
      }
    },
  }));

  const loadedObjRef = useRef(null);
  const controlsRef = useRef(null);

  const isWireframe = wireframeMode !== undefined ? wireframeMode : localWireframe;

  const clippingPlanes = useMemo(() => {
    if (clippingY >= 2) return [];
    return [new THREE.Plane(new THREE.Vector3(0, -1, 0), clippingY)];
  }, [clippingY]);

  // Preload the OBJ once and compute shared transforms
  // This ensures ClickableModel and HeatmapMesh use identical centering/scaling.
  const [meshTransforms, setMeshTransforms] = useState(null);
  const transformsUrlRef = useRef(null);

  useEffect(() => {
    if (!originalObjUrl || transformsUrlRef.current === originalObjUrl) return;
    transformsUrlRef.current = originalObjUrl;

    const loader = new OBJLoader();
    loader.load(originalObjUrl, (obj) => {
      const transforms = computeMeshTransforms(obj);
      setMeshTransforms(transforms);
    });
  }, [originalObjUrl]);

  // Determine visibility: both meshes always mounted, toggle visibility
  const showHeatmap = viewMode === 'heatmap' && !!heatmapData;
  const showClickable = !showHeatmap; // show normal mesh when NOT in heatmap view (or heatmap data not ready)

  const handleMeasureClick = useCallback((point) => {
    setMeasurementPoints(prev => {
      if (prev.length >= 2) return [point];
      return [...prev, point];
    });
  }, []);

  useEffect(() => {
    if (measurementPoints.length === 2) {
      const distance = measurementPoints[0].distanceTo(measurementPoints[1]) * 10;
      onMeasurementUpdate?.(distance);
    } else if (measurementPoints.length < 2) {
      onMeasurementUpdate?.(null);
    }
  }, [measurementPoints, onMeasurementUpdate]);

  const handleClearMeasurement = useCallback(() => {
    setMeasurementPoints([]);
  }, []);

  const handleError = (err) => {
    console.error('Error loading 3D model:', err);
    setError('Failed to load 3D model. Please check the file format.');
  };

  // Camera reset key: changes when URL changes or when user clicks "Reset View"
  const cameraResetKey = `${originalObjUrl}_${resetViewKey}`;

  if (error) {
    return (
      <div className="w-full h-full flex items-center justify-center" style={{ background: '#0F1117' }}>
        <div className="text-center p-8">
          <svg className="w-16 h-16 mx-auto mb-4" style={{ color: '#F87171' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <p style={{ color: '#F87171', fontWeight: 500 }}>{error}</p>
        </div>
      </div>
    );
  }

  if (!originalObjUrl) {
    return (
      <div className="w-full h-full flex items-center justify-center" style={{ background: '#0F1117' }}>
        <div className="text-center p-8">
          <svg className="w-20 h-20 mx-auto mb-6" style={{ color: '#475569' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
          </svg>
          <p style={{ color: '#94A3B8', fontSize: 18, fontWeight: 500, marginBottom: 8 }}>No 3D Model Loaded</p>
          <p style={{ color: '#64748B', fontSize: 14 }}>Upload an artery scan to visualize</p>
        </div>
      </div>
    );
  }

  return (
    <div ref={canvasContainerRef} className="w-full h-full relative overflow-hidden"
      style={{ background: 'linear-gradient(135deg, #0F1117 0%, #141821 100%)' }}>
      <Canvas
        shadows
        gl={{
          antialias: true,
          alpha: true,
          localClippingEnabled: true,
          preserveDrawingBuffer: true,
        }}
        onError={handleError}
      >
        {/* Orthographic Camera */}
        <OrthographicCamera
          makeDefault
          zoom={100}
          position={[3, 2, 3]}
          near={0.1}
          far={1000}
        />

        {/* Camera Controller — auto-frames mesh on load and on reset */}
        <CameraController
          transforms={meshTransforms}
          resetKey={cameraResetKey}
          controlsRef={controlsRef}
        />

        {/* Lighting */}
        <ambientLight intensity={0.6} />
        <directionalLight
          position={[10, 10, 5]}
          intensity={0.9}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
        />
        <directionalLight position={[-5, 5, -5]} intensity={0.3} />
        <pointLight position={[0, 5, 0]} intensity={0.15} color="#4A9EFF" />

        <Environment preset="city" />

        {/* Grid */}
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

        <ContactShadows
          position={[0, -1.49, 0]}
          opacity={0.3}
          scale={10}
          blur={2.5}
          far={3}
          color="#000000"
        />

        <ClippingPlaneHelper clippingY={clippingY} visible={clippingY < 2} />

        {/* BOTH meshes always mounted — visibility toggled, never unmounted */}
        {meshTransforms && (
          <>
            <ClickableModel
              url={originalObjUrl}
              color="#67B8D6"
              opacity={activeHighlight ? 0.25 : 1.0}
              wireframe={isWireframe}
              visible={showClickable}
              clippingPlanes={clippingPlanes}
              isMeasuring={isMeasuring}
              onMeasureClick={handleMeasureClick}
              objRef={loadedObjRef}
              transforms={meshTransforms}
            />

            <Suspense fallback={null}>
              <HeatmapMesh
                url={originalObjUrl}
                heatmapData={heatmapData}
                clippingPlanes={clippingPlanes}
                transforms={meshTransforms}
                visible={showHeatmap}
              />
            </Suspense>
          </>
        )}

        {/* Measurement Highlight Overlay */}
        {activeHighlight && meshTransforms && (
          <MeasurementHighlight
            highlight={activeHighlight}
            transforms={meshTransforms}
          />
        )}

        {/* Camera zoom to highlighted region */}
        {meshTransforms && (
          <HighlightZoomController
            activeHighlight={activeHighlight}
            transforms={meshTransforms}
            controlsRef={controlsRef}
          />
        )}

        {/* Manual Measurement Tool */}
        <MeasurementTool
          points={measurementPoints}
          onClearMeasurement={handleClearMeasurement}
        />

        {/* Controls — completely independent from mesh state */}
        <OrbitControls
          ref={controlsRef}
          makeDefault
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          autoRotate={false}
          minZoom={20}
          maxZoom={500}
          enableDamping={true}
          dampingFactor={0.05}
          zoomToCursor={true}
          minDistance={2}
          maxDistance={15}
        />

        <GizmoHelper
          alignment="bottom-right"
          margin={[100, 140]}
        >
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

      {/* Heatmap Legend */}
      {viewMode === 'heatmap' && heatmapData && (
        <div className="absolute bottom-8 left-8">
          <HeatmapLegend />
        </div>
      )}
    </div>
  );
});

export default ArteryViewer;
