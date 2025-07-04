'use client'

import { motion } from 'framer-motion'
import { FiGithub, FiTwitter, FiMail } from 'react-icons/fi'
import Link from 'next/link'

export default function Footer() {
  return (
    <motion.footer 
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      transition={{ duration: 0.8 }}
      className="bg-dark-200 py-8 mt-20"
    >
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <Link href="/" className="text-xl font-bold flex items-center">
              <img 
                src="./logo.png" 
                alt="Logo" 
                className="h-8 w-8 mr-2" 
              />
              DeepfakeDetector
            </Link>
            <p className="text-gray-400 mt-2">Fighting misinformation with AI</p>
          </div>
          
          <div className="flex space-x-6">
            <motion.a 
              href="https://github.com/yourusername" 
              target="_blank"
              rel="noopener noreferrer"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              aria-label="GitHub"
            >
              <FiGithub className="text-2xl text-gray-400 hover:text-white transition-colors" />
            </motion.a>
            <motion.a 
              href="https://twitter.com/yourusername" 
              target="_blank"
              rel="noopener noreferrer"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              aria-label="Twitter"
            >
              <FiTwitter className="text-2xl text-gray-400 hover:text-white transition-colors" />
            </motion.a>
            <motion.a 
              href="mailto:contact@example.com"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              aria-label="Email"
            >
              <FiMail className="text-2xl text-gray-400 hover:text-white transition-colors" />
            </motion.a>
          </div>
        </div>
        
        <div className="border-t border-dark-400 mt-8 pt-6 text-center text-gray-500">
          <p>© {new Date().getFullYear()} DeepfakeDetector. All rights reserved.</p>
          <div className="flex justify-center space-x-4 mt-2 text-sm">
            <Link href="/privacy" className="hover:text-gray-300">
              Privacy Policy
            </Link>
            <Link href="/terms" className="hover:text-gray-300">
              Terms of Service
            </Link>
          </div>
        </div>
      </div>
    </motion.footer>
  )
}
