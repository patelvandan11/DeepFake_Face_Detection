'use client'

import { motion, AnimatePresence } from 'framer-motion'
import Link from 'next/link'
import { useState } from 'react'
import { FiCamera, FiHome, FiInfo, FiUpload, FiMenu, FiX } from 'react-icons/fi'
import NextImage from 'next/image';

export default function FloatingNavbar() {
  const [activeLink, setActiveLink] = useState('home')
  const [menuOpen, setMenuOpen] = useState(false)

  const navLinks = [
    { name: 'Home', path: '/', icon: <FiHome /> },
    { name: 'Scan', path: '/scan', icon: <FiCamera /> },
    { name: 'Upload', path: '/upload', icon: <FiUpload /> },
    { name: 'About', path: '/about', icon: <FiInfo /> },
  ]

  return (
    <motion.nav
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 20, opacity: 1 }}
      transition={{ type: 'spring', damping: 10 }}
      className="fixed top-0 left-1/2 -translate-x-1/2 z-50 bg-gray-900/80 backdrop-blur-md rounded-3xl shadow-lg border border-gray-700/50 px-6 py-3 md:w-[1000px] lg:w-[r1200px]"
    >
      <div className="flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center mr-8">
          <div className="h-12 w-12 mr-2">
            <NextImage src="/logo.png" alt="Logo" width={48} height={48} className="h-full w-full object-cover" />
          </div>
          <span className="text-xl font-bold text-white">Unmasked</span>
        </div>



        {/* Desktop Navigation */}
        <div className="hidden md:flex space-x-8">
          {navLinks.map((link) => (
            <Link href={link.path} key={link.name}>
              <motion.div
                className="flex flex-col items-center"
                onClick={() => setActiveLink(link.name)}
                whileHover={{ y: -2 }}
              >
                <div className="flex items-center">
                  <span className="mr-2 text-gray-300">{link.icon}</span>
                  <span className={`font-medium ${
                    activeLink === link.name 
                      ? 'text-white font-bold' 
                      : 'text-gray-400 hover:text-white'
                  }`}>
                    {link.name}
                  </span>
                </div>
                
                {activeLink === link.name && (
                  <motion.div 
                    className="h-0.5 w-full bg-cyan-400 mt-1"
                    layoutId="underline"
                    transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
                  />
                )}
              </motion.div>
            </Link>
          ))}
        </div>

        {/* Mobile Menu Button */}
        <button 
          className="md:hidden p-2 text-gray-300 hover:text-white"
          onClick={() => setMenuOpen(!menuOpen)}
        >
          {menuOpen ? <FiX size={20} /> : <FiMenu size={20} />}
        </button>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {menuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden mt-4 space-y-2 overflow-hidden rounded-b-none"
          >
            {navLinks.map((link) => (
              <Link href={link.path} key={`mobile-${link.name}`}>
                <motion.div
                  className={`px-4 py-3 rounded-lg flex items-center ${
                    activeLink === link.name
                      ? 'text-white font-bold bg-gray-800'
                      : 'text-gray-400 hover:text-white'
                  }`}
                  onClick={() => {
                    setActiveLink(link.name)
                    setMenuOpen(false)
                  }}
                >
                  <span className="mr-3">{link.icon}</span>
                  <span>{link.name}</span>
                  {activeLink === link.name && (
                    <motion.div 
                      className="ml-auto h-0.5 w-4 bg-cyan-400"
                      layoutId="mobile-underline"
                    />
                  )}
                </motion.div>
              </Link>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  )
}
