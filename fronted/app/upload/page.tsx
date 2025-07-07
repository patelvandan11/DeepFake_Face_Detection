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
  const [result, setResult] = useState<string | null>(null)

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

  const simulateUpload = async (file: File) => {
    setSelectedOption('Device')
    setUploadProgress(0)

    const formData = new FormData()
    formData.append('file', file)

    const interval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 90) {
          clearInterval(interval)
          return 90
        }
        return prev + 10
      })
    }, 200)

    try {
      const res = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData
      })

      clearInterval(interval)
      setUploadProgress(100)

      if (!res.ok) {
        const err = await res.json()
        alert("Error: " + err.error)
        return
      }

      const data = await res.json()
      setResult(data.prediction) // "Real" or "Fake"
    } catch (err) {
      clearInterval(interval)
      setUploadProgress(100)
      alert("Upload failed: " + (err as any).message)
    }
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
          Analyze videos to detect deepfake manipulation
        </p>

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
                            whileHover={{ x: 5, backgroundColor: 'rgba(30, 30, 30, 0.8)' }}
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
                    setResult(null)
                  }}
                  className="text-gray-400 hover:text-white"
                >
                  <FiX />
                </button>
              </div>

              {selectedOption === 'Device' && (
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
                    {uploadProgress < 100 ? 'Uploading...' : 'Analyzing...'}
                  </p>
                  {result && (
                    <div className="p-4 bg-dark-400 border border-gray-700 rounded-lg">
                      <p className="text-white font-semibold text-lg">
                        Prediction: <span className={result === 'Fake' ? 'text-red-500' : 'text-green-400'}>{result}</span>
                      </p>
                    </div>
                  )}
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
      </motion.div>
    </div>
  )
}
