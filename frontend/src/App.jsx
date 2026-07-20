import { useEffect, useMemo, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Box, Sphere, Grid, Text, Line, Cylinder } from '@react-three/drei'
import './App.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const taskColors = {
  Att: '#ff9f43',
  Rec: '#54a0ff',
  Int: '#ee5253',
  Def: '#a55eea',
}

function visualOffset(id) {
  // Spread agents that share a spawn point so the fleet is visible.
  return [(id % 4) * 22 - 33, Math.floor(id / 4) * 22 - 22]
}

function UAV({ agent, task }) {
  const failed = agent.state === -1
  const committed = agent.commit_until > 0
  const color = failed ? '#6b7280' : agent.type.startsWith('F') ? '#35a7ff' : '#ff5c7a'
  const [ox, oz] = visualOffset(agent.id)
  const x = agent.position[0] + ox
  const z = agent.position[1] + oz
  const route =
    task && task.status !== 2
      ? [
          [x, 8, z],
          [task.position[0], 8, task.position[1]],
        ]
      : null

  return (
    <group>
      {route && <Line points={route} color={committed ? '#ffd166' : '#8ea4ba'} lineWidth={2} />}
      <group position={[x, 18, z]}>
        <Cylinder args={[10, 14, 18, 8]} castShadow>
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={failed ? 0.05 : 0.35}
            transparent={failed}
            opacity={failed ? 0.4 : 1}
          />
        </Cylinder>
        <Box args={[28, 4, 8]} position={[0, 2, 0]}>
          <meshStandardMaterial color={committed ? '#ffd166' : '#dbe7f3'} />
        </Box>
        <Text position={[0, 28, 0]} fontSize={14} color="#ffffff" anchorX="center" outlineWidth={0.5} outlineColor="#000">
          {agent.name}
        </Text>
      </group>
    </group>
  )
}

function TaskObj({ task, time }) {
  const done = task.status === 2
  const unknown = task.known_by === 0
  const overdue = task.deadline != null && time > task.deadline
  const remaining = task.deadline == null ? null : task.deadline - time
  const isEscort = Boolean(task.is_escort || task.kind === 'Escort')
  const color = isEscort ? '#4cc9f0' : overdue ? '#8d99ae' : taskColors[task.type] || '#2ed573'
  const label = isEscort
    ? `Esc${task.id}${task.assigned_agents != null ? ` ${task.assigned_agents}/${task.required_agents || 2}` : ''}`
    : `${task.type}${task.id}${remaining == null ? '' : remaining >= 0 ? ` T-${remaining}` : ' MISS'}`

  return (
    <group position={[task.position[0], 12, task.position[1]]}>
      <Sphere args={[done ? 10 : isEscort ? 12 : 16, 16, 16]}>
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={done || unknown ? 0.05 : 0.45}
          wireframe={!done}
          transparent
          opacity={unknown ? 0.35 : done ? 0.45 : 1}
        />
      </Sphere>
      <Text position={[0, 28, 0]} fontSize={12} color="#ffffff" anchorX="center" outlineWidth={0.4} outlineColor="#000">
        {label}
      </Text>
    </group>
  )
}

function EscortLink({ task }) {
  if (!task.is_escort || task.status === 2 || !task.protected_position) return null
  return (
    <Line
      points={[
        [task.position[0], 18, task.position[1]],
        [task.protected_position[0], 18, task.protected_position[1]],
      ]}
      color="#4cc9f0"
      lineWidth={2}
      dashed
      dashSize={8}
      gapSize={6}
    />
  )
}

function ThreatObj({ threat }) {
  if (threat.status === 0) return null
  return (
    <group position={[threat.position[0], 24, threat.position[1]]}>
      <Sphere args={[14, 8, 8]}>
        <meshStandardMaterial color="#ff1744" emissive="#ff1744" emissiveIntensity={0.6} />
      </Sphere>
      <Text position={[0, 26, 0]} fontSize={12} color="#ffb4c0" anchorX="center" outlineWidth={0.4} outlineColor="#000">
        {`THREAT ${threat.id}`}
      </Text>
    </group>
  )
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function Scene({ frame, tasksById, areaWidth, areaHeight, centerX, centerZ }) {
  return (
    <>
      <color attach="background" args={['#07111e']} />
      <ambientLight intensity={1.1} />
      <directionalLight position={[centerX, 1200, centerZ]} intensity={1.4} />
      <hemisphereLight args={['#9ec9ff', '#0b1724', 0.55]} />

      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[centerX, -1, centerZ]} receiveShadow>
        <planeGeometry args={[areaWidth + 80, areaHeight + 80]} />
        <meshStandardMaterial color="#0c1c2e" />
      </mesh>

      <Grid
        position={[centerX, 0.2, centerZ]}
        args={[areaWidth, areaHeight]}
        cellSize={25}
        cellThickness={0.6}
        cellColor="#1d3550"
        sectionSize={100}
        sectionThickness={1.2}
        sectionColor="#3b6d96"
        fadeDistance={2500}
        fadeStrength={0.4}
        infiniteGrid={false}
      />

      {/* Dual-front divider */}
      <Line points={[[centerX, 2, 0], [centerX, 2, areaHeight]]} color="#ffcf5c" lineWidth={3} />

      {/* Map border */}
      <Line
        points={[
          [0, 2, 0],
          [areaWidth, 2, 0],
          [areaWidth, 2, areaHeight],
          [0, 2, areaHeight],
          [0, 2, 0],
        ]}
        color="#6aa8d8"
        lineWidth={2}
      />

      {frame.agents.map((agent) => (
        <UAV key={agent.name} agent={agent} task={tasksById.get(agent.task_id)} />
      ))}
      {frame.tasks.map((task) => (
        <EscortLink key={`link-${task.type}-${task.id}`} task={task} />
      ))}
      {frame.tasks.map((task) => (
        <TaskObj key={`${task.type}-${task.id}`} task={task} time={frame.time} />
      ))}
      {frame.threats.map((threat) => (
        <ThreatObj key={threat.id} threat={threat} />
      ))}

      <OrbitControls
        makeDefault
        target={[centerX, 0, centerZ]}
        maxPolarAngle={Math.PI / 2.1}
        minDistance={200}
        maxDistance={2500}
      />
    </>
  )
}

function App() {
  const [replay, setReplay] = useState(null)
  const [frameIndex, setFrameIndex] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState(1)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch(`${API}/api/replay`)
      .then((response) => {
        if (!response.ok) throw new Error(`Replay API returned ${response.status}`)
        return response.json()
      })
      .then(setReplay)
      .catch((reason) => setError(reason.message))
  }, [])

  useEffect(() => {
    if (!playing || !replay) return undefined
    const timer = setInterval(() => {
      setFrameIndex((current) => {
        if (current >= replay.frames.length - 1) {
          setPlaying(false)
          return current
        }
        return current + 1
      })
    }, 180 / speed)
    return () => clearInterval(timer)
  }, [playing, replay, speed])

  const frame = replay?.frames[frameIndex]
  const tasksById = useMemo(() => new Map((frame?.tasks || []).map((task) => [task.id, task])), [frame])
  const visibleEvents = useMemo(
    () => (replay?.events || []).filter((event) => event.time <= (frame?.time || 0)).slice(-12).reverse(),
    [replay, frame],
  )

  if (error) {
    return (
      <main className="center">
        <h2>Replay unavailable</h2>
        <p>{error}</p>
        <code>python experiments/generate_simulation_replay.py</code>
      </main>
    )
  }
  if (!replay || !frame) {
    return (
      <main className="center">
        <h2>Loading WPS replay…</h2>
      </main>
    )
  }

  const dynamics = replay.metadata.dynamics
  const metrics = frame.metrics
  const [areaWidth, areaHeight] = replay.metadata.area
  const centerX = areaWidth / 2
  const centerZ = areaHeight / 2

  return (
    <main className="app">
      <header className="topbar">
        <div>
          <div className="eyebrow">
            {replay.metadata.scenario} · seed {replay.metadata.seed}
          </div>
          <h1>{replay.metadata.title}</h1>
          <p>{replay.metadata.algorithm}</p>
        </div>
        <a className="download" href={`${API}/api/replay/download`}>
          Download JSON log
        </a>
      </header>

      <section className="stage">
        <div className="hud left">
          <h3>Mission state</h3>
          <div className="metrics">
            <Metric label="Step" value={`${frame.time}/${replay.metadata.max_time_steps}`} />
            <Metric label="S_WPS" value={(metrics.s_wps ?? 0).toFixed(1)} />
            {metrics.s_esc != null && <Metric label="S_ESC" value={metrics.s_esc.toFixed(1)} />}
            <Metric label="Agents" value={metrics.active_agents} />
            <Metric label="Open tasks" value={metrics.open_tasks} />
            <Metric label="On time" value={metrics.on_time} />
            <Metric label="Missed" value={metrics.missed} />
            <Metric label="Switches" value={metrics.switches} />
            {metrics.escort_coverage != null && (
              <Metric label="Escort cov" value={`${(100 * metrics.escort_coverage).toFixed(0)}%`} />
            )}
            {metrics.recon_losses != null && <Metric label="Rec losses" value={metrics.recon_losses} />}
          </div>
          <h3>Dynamics enabled</h3>
          <div className="chips">
            <span>dual fronts</span>
            <span>burst ×{dynamics.burst_size}</span>
            <span>arrival {dynamics.arrival_rate}</span>
            <span>failure {dynamics.fail_rate}</span>
            <span>sense {dynamics.sense_radius}</span>
            <span>delay {dynamics.threat_delay}</span>
            <span>window {dynamics.window_length}</span>
            <span>commit {dynamics.commit_horizon}</span>
          </div>
          <div className="legend">
            <span>
              <i className="fighter" /> Fighter
            </span>
            <span>
              <i className="recon" /> Recon
            </span>
            <span>
              <i className="dynamic" /> Dynamic task
            </span>
            <span>
              <i className="threat" /> Threat
            </span>
          </div>
        </div>

        <div className="hud right">
          <h3>Event log</h3>
          <div className="events">
            {visibleEvents.length === 0 && <p className="muted">No dynamic events yet.</p>}
            {visibleEvents.map((event, index) => (
              <div className="event" key={`${event.time}-${event.type}-${index}`}>
                <time>T{event.time}</time>
                <div>
                  <strong>{event.type}</strong>
                  <small>{event.detail.join(' · ')}</small>
                </div>
              </div>
            ))}
          </div>
        </div>

        <Canvas
          camera={{
            position: [centerX, 780, centerZ + 920],
            fov: 42,
            near: 1,
            far: 6000,
          }}
        >
          <Scene
            frame={frame}
            tasksById={tasksById}
            areaWidth={areaWidth}
            areaHeight={areaHeight}
            centerX={centerX}
            centerZ={centerZ}
          />
        </Canvas>
      </section>

      <footer className="controls">
        <button onClick={() => setPlaying((value) => !value)}>{playing ? 'Pause' : 'Play'}</button>
        <button
          onClick={() => {
            setFrameIndex(0)
            setPlaying(true)
          }}
        >
          Restart
        </button>
        <input
          aria-label="Replay position"
          type="range"
          min="0"
          max={replay.frames.length - 1}
          value={frameIndex}
          onChange={(event) => {
            setFrameIndex(Number(event.target.value))
            setPlaying(false)
          }}
        />
        <select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
          <option value={0.5}>0.5×</option>
          <option value={1}>1×</option>
          <option value={2}>2×</option>
          <option value={4}>4×</option>
        </select>
        <span>
          {frameIndex + 1} / {replay.frames.length} frames
        </span>
      </footer>
    </main>
  )
}

export default App
