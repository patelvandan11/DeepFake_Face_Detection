'use client'
import { useEffect, useState, useCallback, useRef } from 'react'
import { motion } from 'framer-motion'
const GlitterCursor = () => {
  const [particles, setParticles] = useState<Array<{
    id: number
    x: number
    y: number
    size: number
    lifetime: number
    rotation: number
  }>>([])
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 })
  const [isMobile, setIsMobile] = useState(false)
  const particleId = useRef(0)
  const lastPos = useRef({ x: 0, y: 0 })

  // Detect mobile devices
  useEffect(() => {
    const checkMobile = () => {
      // Basic check for mobile devices
      const isTouch =
        typeof window !== 'undefined' &&
        ('ontouchstart' in window || navigator.maxTouchPoints > 0)
      setIsMobile(isTouch || window.innerWidth < 768)
    }
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  // Generate glitter particles
  const handleMouseMove = useCallback((e: MouseEvent) => {
    const currentPos = { x: e.clientX, y: e.clientY }
    setMousePos(currentPos)
    // Calculate distance moved
    const distance = Math.sqrt(
      Math.pow(currentPos.x - lastPos.current.x, 2) +
      Math.pow(currentPos.y - lastPos.current.y, 2)
    )
    // Add particles based on movement distance
    if (distance > 3) {
      const newParticles = Array(Math.floor(distance / 10)).fill(0).map((_, i) => ({
        id: particleId.current++,
        x: lastPos.current.x + (currentPos.x - lastPos.current.x) * (i / Math.floor(distance / 10)),
        y: lastPos.current.y + (currentPos.y - lastPos.current.y) * (i / Math.floor(distance / 10)),
        size: Math.random() * 4 + 2, // Slightly larger particles
        lifetime: Math.random() * 0.8 + 0.5,
        rotation: Math.random() * 360
      }))
      setParticles(prev => [...prev, ...newParticles].slice(-300))
      lastPos.current = currentPos
    }
  }, [])

  // Update particle lifetimes
  useEffect(() => {
    if (isMobile) return
    const interval = setInterval(() => {
      setParticles(prev => prev
        .map(p => ({ ...p, lifetime: p.lifetime - 0.02 }))
        .filter(p => p.lifetime > 0)
      )
    }, 20)
    return () => clearInterval(interval)
  }, [isMobile])

  // Set up event listeners
  useEffect(() => {
    if (isMobile) return
    window.addEventListener('mousemove', handleMouseMove)
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [handleMouseMove, isMobile])

  if (isMobile) return null

  return (
    <>
      {/* Main Cursor */}
      <motion.div
        className="fixed w-4 h-4 rounded-full pointer-events-none z-50"
        style={{
          left: `${mousePos.x}px`,
          top: `${mousePos.y}px`,
          background: 'radial-gradient(circle, #b300ff, #6e00ff)',
          boxShadow: '0 0 25px 10px rgba(180, 0, 255, 0.5)',
          filter: 'blur(0.3px) brightness(1.2)'
        }}
      />

      {/* Glitter Particles */}
      <div className="fixed inset-0 pointer-events-none z-40 overflow-hidden">
        {particles.map((particle) => {
          const progress = 1 - particle.lifetime
          return (
            <motion.div
              key={`glitter-${particle.id}`}
              className="absolute rounded-sm pointer-events-none"
              style={{
                left: `${particle.x}px`,
                top: `${particle.y}px`,
                width: `${particle.size}px`,
                height: `${particle.size}px`,
                background: `radial-gradient(
                  circle, 
                  rgba(255, 255, 255, ${0.8 * (1 - progress)}), 
                  rgba(179, 0, 255, ${0.7 * (1 - progress)})
                )`,
                transform: `translate(-50%, -50%) rotate(${particle.rotation}deg) scale(${0.3 + 0.7 * (1 - progress)})`,
                opacity: 0.7 * (1 - progress),
                boxShadow: `0 0 ${Math.max(2, 5 * progress)}px rgba(180, 0, 255, 0.7)`,
                filter: `blur(${Math.max(0.2, 1 * progress)}px) brightness(${1 + progress})`
              }}
            />
          )
        })}
      </div>
    </>
  )
}

export default GlitterCursor