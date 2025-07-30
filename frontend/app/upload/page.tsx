'use client'

import { motion, AnimatePresence } from 'framer-motion';
import { FiUpload, FiLink, FiHardDrive, FiCloud, FiInstagram, FiChevronDown, FiX, FiYoutube } from 'react-icons/fi';
import { useState, useRef, ChangeEvent, useCallback } from 'react'; // Added useCallback
import { useGoogleLogin } from '@react-oauth/google';

export default function UploadPage() {
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  const [isDropdownOpen, setDropdownOpen] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [result, setResult] = useState<string | null>(null);
  const [urlInput, setUrlInput] = useState<string>('');

  const uploadOptions = [
    { name: 'Device', icon: <FiHardDrive />, color: 'text-blue-400' },
    { name: 'Google Drive', icon: <FiCloud />, color: 'text-green-400' },
    { name: 'Instagram', icon: <FiInstagram />, color: 'text-pink-400' },
    { name: 'YouTube', icon: <FiYoutube />, color: 'text-red-600' }, 
    { name: 'OneDrive', icon: <FiCloud />, color: 'text-blue-500' },
    { name: 'URL', icon: <FiLink />, color: 'text-purple-400' }
  ]

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      simulateUpload(e.target.files[0])
    }
  }

  const simulateUpload = useCallback(async (file: File) => {
    setSelectedOption('Device')
    setUploadProgress(0)
    setResult(null);

    const formData = new FormData()
    formData.append('file', file)

    const apiUrl = process.env.NEXT_PUBLIC_API_URL;
    if (!apiUrl) {
      alert("API URL is not configured. Please contact support.");
      return;
    }

    // This interval is just for visual feedback
    const interval = setInterval(() => {
      setUploadProgress(prev => Math.min(prev + 10, 90));
    }, 200);

    try {
      const res = await fetch(`${apiUrl}/analyze`, {
        method: "POST",
        body: formData
      });

      clearInterval(interval);
      setUploadProgress(100);

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Analysis failed');
      }

      const data = await res.json();
      setResult(data.prediction);
    } catch (err) {
      clearInterval(interval);
      setUploadProgress(0);
      setSelectedOption(null);
      // FIX 2: Use type-safe error handling
      if (err instanceof Error) {
        alert("Upload failed: " + err.message);
      } else {
        alert("An unknown upload error occurred.");
      }
    }
  }, []); // Empty dependency array is fine if it doesn't use other component state/props

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      simulateUpload(e.dataTransfer.files[0]);
    }
  }, [simulateUpload]);

  const handleUrlSubmit = () => {
    if (!urlInput) {
      alert('Please enter a URL');
      return;
    }
    alert(`URL entered: ${urlInput}`);
    // Add your URL processing logic here
  };

  const login = useGoogleLogin({
    flow: 'auth-code', // This is crucial for backend access
    scope: 'https://www.googleapis.com/auth/drive.readonly',
    onSuccess: async (codeResponse) => {
      console.log("Sending auth code to backend:", codeResponse.code);
      try {
        // Send the one-time code to your backend API route
        const res = await fetch('/api/auth/google', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code: codeResponse.code }),
        });
        
        if (res.ok) {
          alert("Successfully connected to Google Drive! You can now implement the file picker.");
          // You can set state here to confirm the user is connected
        } else {
          throw new Error('Failed to connect to Google Drive.');
        }
      } catch (error) {
        console.error("Authentication error:", error);
        alert("Could not connect to Google Drive.");
      }
    },
    onError: (errorResponse) => {
      console.error("Google Login Failed:", errorResponse);
      alert("Google login failed.");
    },
  });

  const handleGoogleDrive = async (accessToken: string) => {
  try {
    setSelectedOption('Google Drive');
    setUploadProgress(0);
    setResult(null);
    
    
    const file = await pickGoogleDriveFile(accessToken);
    if (!file) throw new Error('No file selected');
    
    // Simulate progress
    const interval = setInterval(() => {
      setUploadProgress(prev => Math.min(prev + 10, 90));
    }, 200);

    // Process the file
    // const result = await processFile(file);
    // setResult(result);
    setUploadProgress(100);
    
    clearInterval(interval);
  } catch (error) {
    setUploadProgress(0);
    setSelectedOption(null);
    if (error instanceof Error) {
      setGoogleDriveError(error.message);
    } else {
      setGoogleDriveError('Failed to process Google Drive file');
    }
  } finally {
    setIsGoogleDriveLoading(false);
  }
};
  
  // OneDrive Picker
  const handleOneDrive = () => {
    // Implement OneDrive file picker logic here
    alert("OneDrive file picker is not implemented yet.");
  };

  // Handle cloud files
  const handleCloudFile = async (source: string, fileIdentifier: string) => {
    setSelectedOption(source)
    setUploadProgress(0)
    
    try {
      const response = await fetch('/api/download', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          source,
          fileIdentifier
        }),
      })

      const data = await response.json()
      setResult(data.prediction)
      setUploadProgress(100)
    } catch (error) {
      console.error('Upload failed:', error)
      alert(`Failed to process ${source} file`)
    }
  }

  // Update your option handler
  const handleOptionSelect = (optionName: string) => {
    setDropdownOpen(false);
    switch(optionName) {
      case 'Device':
        fileInputRef.current?.click();
        break;
      case 'Google Drive':
        handleGoogleDrive();
        break;
      case 'OneDrive':
        handleOneDrive();
        break;
      case 'URL':
        setSelectedOption('URL');
        break;
      default:
        setSelectedOption(optionName);
    }
  };
  
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
                    setUrlInput('')
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

              {(selectedOption === 'YouTube' || selectedOption === 'Instagram' || selectedOption === 'URL') && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4 mt-4">
                  <input
                    type="text"
                    value={urlInput}
                    onChange={e => setUrlInput(e.target.value)}
                    placeholder={`Paste ${selectedOption} video URL here`}
                    className="w-full px-4 py-2 rounded-md bg-gray-800 border border-gray-600 text-white placeholder-gray-500"
                  />
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="px-4 py-2 rounded-md bg-blue-600 text-white font-medium"
                    onClick={handleUrlSubmit}
                  >
                    Submit URL
                  </motion.button>
                </motion.div>
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
