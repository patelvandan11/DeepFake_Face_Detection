'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { FiUpload, FiLink, FiCamera, FiHardDrive, FiCloud, FiInstagram, FiChevronDown, FiX } from 'react-icons/fi'
import { useState, useRef, ChangeEvent } from 'react'

export default function UploadPage() {
  const [selectedOption, setSelectedOption] = useState<string | null>(null)
  const [isDropdownOpen, setDropdownOpen] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [showCamera, setShowCamera] = useState(false)
  const [showUrlModal, setShowUrlModal] = useState(false)
  const connectGoogleDrive = () => {
    window.open('https://accounts.google.com/...', '_blank')
  }

  const uploadOptions = [
    { name: 'Device', icon: <FiHardDrive />, color: 'text-blue-400' },
    { name: 'Google Drive', icon: <FiCloud />, color: 'text-green-400' },
    { name: 'Instagram', icon: <FiInstagram />, color: 'text-pink-400' },
    { name: 'OneDrive', icon: <FiCloud />, color: 'text-blue-500' },
    { name: 'URL', icon: <FiLink />, color: 'text-purple-400' }
  ]

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      simulateUpload(e.target.files[0])
    }
  }

  const simulateUpload = (file: File) => {
    setSelectedOption('Device')
    setUploadProgress(0)
    const interval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval)
          return 100
        }
        return prev + 10
      })
    }, 300)
  }

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      simulateUpload(e.dataTransfer.files[0])
    }
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-3xl mx-auto"
      >
        <h1 className="text-4xl md:text-5xl font-bold mb-6 text-center bg-gradient-to-r from-purple-400 to-blue-500 bg-clip-text text-transparent">
          Upload Media
        </h1>
        <p className="text-gray-400 text-center mb-10 max-w-lg mx-auto">
          Analyze images from multiple sources to detect deepfake manipulation
        </p>

        {/* Main Upload Card */}
        <motion.div 
          className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all ${dragActive ? 'border-blue-500 bg-dark-300' : 'border-gray-700 bg-dark-200'}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          whileHover={{ scale: 1.01 }}
        >
          {!selectedOption ? (
            <>
              <div className="flex justify-center mb-6">
          <div className="relative">
              <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="flex items-center px-6 py-3 bg-gradient-to-r from-purple-500 to-blue-600 rounded-lg font-medium"
            onClick={() => setDropdownOpen(!isDropdownOpen)}
              >
            <FiUpload className="mr-2" />
            Select Source
            <FiChevronDown className={`ml-2 transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`} />
              </motion.button>

              <AnimatePresence>
            {isDropdownOpen && (
            <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute left-0 right-0 mt-2 bg-black rounded-lg shadow-xl z-10 overflow-hidden border border-gray-800"
            >
                {uploadOptions.map((option) => (
                <motion.button
              key={option.name}
              whileHover={{ 
              x: 5,
              backgroundColor: 'rgba(30, 30, 30, 0.8)'
              }}
              className={`flex items-center w-full px-4 py-3 text-left ${option.color} bg-black hover:bg-gray-900 transition-colors`}
              onClick={() => {
              if (option.name === 'Device') {
                  fileInputRef.current?.click()
              }
              setSelectedOption(option.name)
              setDropdownOpen(false)
              }}
                >
              <span className="mr-3">{option.icon}</span>
              {option.name}
                </motion.button>
                ))}
            </motion.div>
            )}
              </AnimatePresence>
          </div>
              </div>

              <p className="text-gray-400 mb-4">or drag and drop video files here</p>
              <p className="text-sm text-gray-500">Supports: MP4, AVI, MOV (Max 100MB)</p>
            </>
          ) : (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-6"
            >
              <div className="flex justify-between items-center">
          <div className="flex items-center">
            {uploadOptions.find(o => o.name === selectedOption)?.icon}
            <span className="ml-2">{selectedOption}</span>
          </div>
          <button 
            onClick={() => {
              setSelectedOption(null)
              setUploadProgress(0)
            }}
            className="text-gray-400 hover:text-white"
          >
            <FiX />
          </button>
              </div>

              {selectedOption === 'Device' ? (
          <div className="space-y-4">
            <div className="h-2.5 bg-dark-300 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-purple-500 to-blue-500"
                initial={{ width: 0 }}
                animate={{ width: `${uploadProgress}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
            <p className="text-sm text-gray-400">
              {uploadProgress < 100 ? 'Uploading...' : 'Analysis starting...'}
            </p>
          </div>
              ) : (
          <div className="p-6 bg-dark-300 rounded-lg">
            <p className="text-gray-400">
              {`Connect to ${selectedOption} to select files`}
            </p>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="mt-4 px-4 py-2 bg-dark-400 rounded-lg"
            >
              Connect {selectedOption}
            </motion.button>
          </div>
              )}
            </motion.div>
          )}

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="video/mp4,video/x-m4v,video/*"
            className="hidden"
          />
        </motion.div>

        {/* Recent Uploads Section */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="mt-12"
        >
          <h3 className="text-xl font-bold mb-4 flex items-center">
            <FiCamera className="mr-2 text-blue-400" />
            Recent Uploads
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
            {[1, 2, 3, 4].map((item) => (
              <motion.div
                key={item}
                whileHover={{ y: -5 }}
                className="aspect-square bg-dark-300 rounded-lg overflow-hidden cursor-pointer"
              >
                <div className="w-full h-full bg-dark-400 flex items-center justify-center text-gray-500">
                  Thumbnail {item}
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Camera Capture Option */}
        <motion.div 
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="mt-12 p-6 bg-gradient-to-br from-dark-300 to-dark-400 rounded-xl border border-gray-700"
        >
          <div className="flex flex-col md:flex-row items-center">
            <div className="flex-1 mb-4 md:mb-0 md:mr-6">
              <h3 className="text-xl font-bold mb-2">Capture Live Image</h3>
              <p className="text-gray-400">
                Use your camera to instantly analyze faces in real-time
              </p>
            </div>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-6 py-3 bg-blue-500/20 border border-blue-500 rounded-lg font-medium flex items-center"
            >
              <FiCamera className="mr-2" />
              Open Camera
            </motion.button>
          </div>
        </motion.div>
      </motion.div>
    </div>
  )
}