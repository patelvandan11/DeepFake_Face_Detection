'use client'

import { motion } from 'framer-motion'
import { FiCheck, FiX, FiAlertTriangle, FiInfo, FiBarChart2 } from 'react-icons/fi'

interface DetectionDetails {
  faceAnalysis: string
  artifacts: string
  blending: string
  resolution?: string
  predictionScore?: number
}

interface DetectionResult {
  isDeepfake: boolean
  confidence: number
  details: DetectionDetails
  timestamp?: string
}

interface ResultsDisplayProps {
  results: DetectionResult | null
  isLoading: boolean
  error?: string | null
}

export default function ResultsDisplay({ results, isLoading, error }: ResultsDisplayProps) {
  if (isLoading) {
    return (
      <div className="bg-dark-200 rounded-xl p-6 shadow-lg h-full flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Analyzing image for deepfake indicators...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="bg-dark-200 rounded-xl p-6 shadow-lg h-full flex items-center justify-center"
      >
        <div className="text-center text-red-400">
          <FiAlertTriangle className="text-4xl mx-auto mb-4" />
          <h3 className="text-xl font-bold mb-2">Analysis Failed</h3>
          <p className="text-red-300">{error}</p>
          <p className="text-sm text-gray-500 mt-4">Please try again or use a different image</p>
        </div>
      </motion.div>
    )
  }

  if (!results) {
    return (
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="bg-dark-200 rounded-xl p-6 shadow-lg h-full flex items-center justify-center"
      >
        <div className="text-center">
          <FiInfo className="text-4xl text-gray-500 mx-auto mb-4" />
          <h3 className="text-xl font-bold mb-2">No Results Yet</h3>
          <p className="text-gray-400">Upload an image to analyze for deepfake detection</p>
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="bg-dark-200 rounded-xl p-6 shadow-lg h-full overflow-y-auto"
    >
      <h2 className="text-2xl font-bold mb-6">Detection Results</h2>
      
      {/* Main Result Card */}
      <div className="mb-8 bg-dark-300 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-medium">Deepfake Prediction</h3>
          <div className={`flex items-center ${results.isDeepfake ? 'text-red-400' : 'text-green-400'}`}>
            {results.isDeepfake ? (
              <>
                <FiX className="mr-1.5" />
                <span>Likely Deepfake</span>
              </>
            ) : (
              <>
                <FiCheck className="mr-1.5" />
                <span>Likely Authentic</span>
              </>
            )}
          </div>
        </div>
        
        {/* Confidence Meter */}
        <div className="mb-2">
          <div className="flex justify-between text-sm text-gray-400 mb-1">
            <span>0%</span>
            <span>Confidence: {results.confidence.toFixed(1)}%</span>
            <span>100%</span>
          </div>
          <div className="w-full bg-dark-400 rounded-full h-2.5">
            <motion.div 
              className={`h-2.5 rounded-full ${results.isDeepfake ? 'bg-red-500' : 'bg-green-500'}`}
              initial={{ width: 0 }}
              animate={{ width: `${results.confidence}%` }}
              transition={{ duration: 1, type: 'spring' }}
            />
          </div>
        </div>
        
        {results.timestamp && (
          <p className="text-xs text-gray-500 text-right mt-1">
            Analyzed at {new Date(results.timestamp).toLocaleString()}
          </p>
        )}
      </div>
      
      {/* Detailed Analysis */}
      <div className="space-y-5">
        <h3 className="text-lg font-medium flex items-center">
          <FiBarChart2 className="mr-2" />
          Detailed Analysis
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(results.details).map(([key, value]) => (
            <motion.div
              key={key}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * Object.keys(results.details).indexOf(key) }}
              className="bg-dark-300 p-4 rounded-lg"
            >
              <h4 className="text-sm text-gray-400 mb-1.5 capitalize">
                {key.replace(/([A-Z])/g, ' $1').trim()}
              </h4>
              <p className={`font-medium ${
                (key === 'artifacts' || key === 'blending') && 
                (value.toLowerCase().includes('detect') || value.toLowerCase().includes('suspicious')) 
                  ? 'text-yellow-400' 
                  : ''
              }`}>
                {value}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
      
      {/* Warning/Info Box */}
      {results.isDeepfake ? (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg flex items-start"
        >
          <FiAlertTriangle className="text-red-400 mr-3 mt-0.5 flex-shrink-0" />
          <div>
            <h4 className="font-bold text-red-400 mb-1">Warning</h4>
            <p className="text-sm text-red-300">
              This image shows signs of digital manipulation. Deepfakes can be used to spread misinformation.
            </p>
          </div>
        </motion.div>
      ) : (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-6 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg flex items-start"
        >
          <FiCheck className="text-blue-400 mr-3 mt-0.5 flex-shrink-0" />
          <div>
            <h4 className="font-bold text-blue-400 mb-1">Authentic Content</h4>
            <p className="text-sm text-blue-300">
              No significant signs of digital manipulation detected. However, some sophisticated deepfakes may evade detection.
            </p>
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}