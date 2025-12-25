"use client"
import { motion } from 'framer-motion'
import Link from 'next/link'

export default function Home() {
  return (
    <div className="container mx-auto px-4">
      <motion.section 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8 }}
        className="text-center py-20"
      >
        <motion.h1 
          className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent"
          initial={{ y: -50 }}
          animate={{ y: 0 }}
          transition={{ duration: 0.5 }}
        >
          Detect Deepfake Faces
        </motion.h1>
        
        <motion.p 
          className="text-xl md:text-2xl text-gray-400 max-w-3xl mx-auto mb-10"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.8 }}
        >
          Our advanced AI model helps you identify manipulated media with high accuracy
        </motion.p>
        
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6, duration: 0.8 }}
        >
          <Link href="/detect">
            <motion.button
              className="px-8 py-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg font-bold text-lg shadow-lg hover:shadow-xl transition-all"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Try It Now
            </motion.button>
          </Link>
        </motion.div>
      </motion.section>

      <motion.section 
        className="mt-20 grid md:grid-cols-3 gap-8"
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        transition={{ duration: 0.8 }}
        viewport={{ once: true }}
      >
        {[
          {
            title: "Real-time Detection",
            desc: "Instant analysis of uploaded images with detailed results",
            icon: "⚡"
          },
          {
            title: "Advanced AI",
            desc: "Powered by state-of-the-art deep learning models",
            icon: "🧠"
          },
          {
            title: "Privacy Focused",
            desc: "Your data never leaves your device",
            icon: "🔒"
          }
        ].map((feature, i) => (
          <motion.div 
            key={i}
            className="bg-dark-200 p-6 rounded-xl shadow-lg"
            whileHover={{ y: -10 }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            <div className="text-4xl mb-4">{feature.icon}</div>
            <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
            <p className="text-gray-400">{feature.desc}</p>
          </motion.div>
        ))}
      </motion.section>
    </div>
  )
}