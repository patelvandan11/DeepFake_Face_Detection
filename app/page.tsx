"use client"
import { motion } from 'framer-motion'
import Link from 'next/link'

// Define animation variants for cleaner code
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.2, // Stagger children animations
    },
  },
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: { duration: 0.5 },
  },
};

const features = [
  {
    title: "Real-time Detection",
    desc: "Instant analysis of uploaded images with detailed results.",
    icon: "⚡"
  },
  {
    title: "Advanced AI",
    desc: "Powered by state-of-the-art deep learning models.",
    icon: "🧠"
  },
  {
    title: "Privacy Focused",
    desc: "Your data is processed securely and is not stored.",
    icon: "🔒"
  }
];

export default function Home() {
  return (
    <div className="container mx-auto px-4">
      <motion.section
        className="text-center py-20"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <motion.h1
          className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent"
          variants={itemVariants}
        >
          Detect Deepfake Faces
        </motion.h1>

        <motion.p
          className="text-xl md:text-2xl text-gray-400 max-w-3xl mx-auto mb-10"
          variants={itemVariants}
        >
          Our advanced AI model helps you identify manipulated media with high accuracy.
        </motion.p>

        <motion.div variants={itemVariants}>
          <Link href="/detect" passHref>
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
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, amount: 0.3 }}
      >
        {features.map((feature) => (
          <motion.div
            key={feature.title} // Use a unique string key instead of index
            className="bg-dark-200 p-6 rounded-xl shadow-lg"
            variants={itemVariants}
            whileHover={{ y: -10 }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            <div className="text-4xl mb-4" role="img" aria-label={`${feature.title} icon`}>
              {feature.icon}
            </div>
            <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
            <p className="text-gray-400">{feature.desc}</p>
          </motion.div>
        ))}
      </motion.section>
    </div>
  )
}
