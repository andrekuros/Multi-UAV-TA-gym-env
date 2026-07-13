import { useState, useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Box, Sphere, Grid, Text } from '@react-three/drei'
import './App.css'

function UAV({ agent }) {
  return (
    <group position={[agent.position[0], 20, agent.position[1]]}>
      <Box args={[10, 5, 10]}>
        <meshStandardMaterial color={agent.name.startsWith("F") ? "blue" : "red"} />
      </Box>
      <Text position={[0, 10, 0]} fontSize={10} color="white">
        {agent.name}
      </Text>
    </group>
  )
}

function TaskObj({ task }) {
  return (
    <group position={[task.position[0], 0, task.position[1]]}>
      <Sphere args={[15, 16, 16]}>
        <meshStandardMaterial color="green" wireframe />
      </Sphere>
      <Text position={[0, 25, 0]} fontSize={10} color="white">
        Task {task.id}
      </Text>
    </group>
  )
}

function App() {
  const [gameState, setGameState] = useState({ agents: [], tasks: [], time_steps: 0 })

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/simulation')
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setGameState(data)
    }

    return () => ws.close()
  }, [])

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#1a1a1a' }}>
      <div style={{ position: 'absolute', top: 20, left: 20, color: 'white', zIndex: 10, fontFamily: 'monospace' }}>
        <h2>Multi-UAV-TA Simulation</h2>
        <p>Time Steps: {gameState.time_steps}</p>
        <p>Active Agents: {gameState.agents.length}</p>
        <p>Active Tasks: {gameState.tasks.length}</p>
      </div>

      <Canvas camera={{ position: [500, 1000, 1000], fov: 60 }}>
        <color attach="background" args={['#1a1a1a']} />
        <ambientLight intensity={0.5} />
        <directionalLight position={[100, 1000, 500]} intensity={1} />
        
        {/* Render Ground */}
        <Grid infiniteGrid fadeDistance={4000} sectionColor="#444" cellColor="#222" />

        {/* Render Agents */}
        {gameState.agents.map(agent => (
          <UAV key={agent.name} agent={agent} />
        ))}

        {/* Render Tasks */}
        {gameState.tasks.map(task => (
          <TaskObj key={task.id} task={task} />
        ))}

        <OrbitControls target={[500, 0, 500]} />
      </Canvas>
    </div>
  )
}

export default App
