"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { TissueGraph } from "@/lib/tissue";
import { colorFor } from "@/lib/palette";

function Nodes({ graph }: { graph: TissueGraph }) {
  const ref = useRef<THREE.InstancedMesh>(null);

  useEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();
    const color = new THREE.Color();
    graph.cells.forEach((cell, i) => {
      dummy.position.set(...cell.position);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      mesh.setColorAt(i, color.set(colorFor(cell.type)));
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [graph]);

  return (
    <instancedMesh
      ref={ref}
      args={[undefined, undefined, graph.cells.length]}
      frustumCulled={false}
    >
      <sphereGeometry args={[0.13, 18, 18]} />
      <meshStandardMaterial
        roughness={0.35}
        metalness={0.1}
        emissiveIntensity={0.15}
        toneMapped={false}
      />
    </instancedMesh>
  );
}

function Edges({ graph }: { graph: TissueGraph }) {
  const positions = useMemo(() => {
    const arr = new Float32Array(graph.edges.length * 6);
    graph.edges.forEach(([i, j], e) => {
      const a = graph.cells[i].position;
      const b = graph.cells[j].position;
      arr.set([a[0], a[1], a[2], b[0], b[1], b[2]], e * 6);
    });
    return arr;
  }, [graph]);

  return (
    <lineSegments key={positions.length}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <lineBasicMaterial color="#6c9bde" transparent opacity={0.14} toneMapped={false} />
    </lineSegments>
  );
}

export function CellGraph3D({
  graph,
  autoRotate = true,
}: {
  graph: TissueGraph;
  autoRotate?: boolean;
}) {
  const [mounted, setMounted] = useState(false);
  // Mount guard keeps the WebGL canvas out of SSR; setting state on mount is intended.
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => setMounted(true), []);

  if (!mounted) {
    return (
      <div className="grid h-full w-full place-items-center text-sm text-ink-muted">
        Loading 3D scene…
      </div>
    );
  }

  return (
    <Canvas
      camera={{ position: [0, 1, 19], fov: 45 }}
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true }}
      style={{ width: "100%", height: "100%" }}
    >
      <fog attach="fog" args={["#07080b", 20, 38]} />
      <ambientLight intensity={1.2} />
      <pointLight position={[12, 14, 12]} intensity={140} />
      <pointLight position={[-14, -8, -10]} intensity={70} color="#9085e9" />
      <group key={graph.cells.length}>
        <Edges graph={graph} />
        <Nodes graph={graph} />
      </group>
      <OrbitControls
        enablePan={false}
        enableZoom
        autoRotate={autoRotate}
        autoRotateSpeed={0.55}
        enableDamping
        dampingFactor={0.08}
        minDistance={9}
        maxDistance={34}
      />
    </Canvas>
  );
}
