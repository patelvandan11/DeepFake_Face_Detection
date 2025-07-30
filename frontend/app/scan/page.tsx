'use client';

import { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

const videoConstraints = {
  width: 720,
  height: 480,
  facingMode: 'user',
};

const ScanPage = () => {
  const webcamRef = useRef<Webcam>(null);
  const [source, setSource] = useState('custom');
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<{ isFake: boolean; confidence: number } | null>(null);
  const [gridActive, setGridActive] = useState(true);

  // Cyberpunk grid animation effect
  useEffect(() => {
    const interval = setInterval(() => {
      setGridActive(prev => !prev);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleScan = async () => {
    if (!webcamRef.current) return;

    const screenshot = webcamRef.current.getScreenshot();
    if (!screenshot) return alert('Failed to capture image from webcam.');

    setScanning(true);
    setResult(null);

    try {
      const response = await axios.post('/api/detect', {
        image: screenshot,
        source,
      });

      setResult(response.data);
    } catch (error) {
      console.error('Detection failed', error);
      alert('Detection failed. Please try again.');
    } finally {
      setScanning(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col items-center justify-center px-4 py-8 overflow-hidden relative">
      {/* Sci-fi background elements */}
      <div className="absolute inset-0 overflow-hidden opacity-20">
        <div className={`absolute inset-0 transition-opacity duration-1000 ${gridActive ? 'opacity-100' : 'opacity-30'}`}>
          <div className="grid-lines h-full w-full" style={{
            backgroundImage: `
              linear-gradient(rgba(20, 255, 236, 0.3) 1px, transparent 1px),
              linear-gradient(90deg, rgba(20, 255, 236, 0.3) 1px, transparent 1px)
            `,
            backgroundSize: '50px 50px'
          }}></div>
        </div>
        <div className="absolute inset-0 bg-gradient-to-br from-transparent via-blue-900/10 to-transparent"></div>
      </div>

      <div className="w-full max-w-2xl relative z-10">
        {/* Header with sci-fi styling */}
        <div className="text-center mb-8 relative">
          <h1 className="text-5xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-500 tracking-tight">
            Unmasked<span className="text-blue-400">AI</span>
          </h1>
          <p className="text-blue-300 font-mono text-sm tracking-widest">PROCTORING SYSTEM v2.4.7</p>
          <div className="absolute -bottom-4 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-blue-500 to-transparent"></div>
        </div>

        {/* Main scanning panel */}
        <div className="bg-gray-900/80 backdrop-blur-sm border border-gray-700/50 rounded-xl shadow-2xl overflow-hidden">
          {/* Panel header */}
          <div className="bg-gradient-to-r from-gray-800 to-gray-900 border-b border-gray-700/50 p-4 flex items-center">
            <div className="flex space-x-2 mr-4">
              <div className="h-3 w-3 rounded-full bg-red-500"></div>
              <div className="h-3 w-3 rounded-full bg-yellow-500"></div>
              <div className="h-3 w-3 rounded-full bg-green-500"></div>
            </div>
            <div className="font-mono text-sm text-blue-400 tracking-wider">LIVE_FEED_INTERFACE</div>
          </div>

          <div className="p-6">
            {/* Source selection */}
            <div className="mb-6">
              <label className="block text-cyan-400 font-mono text-sm mb-2 tracking-wider">SOURCE SELECTION</label>
              <div className="relative">
                <select
                  value={source}
                  onChange={(e) => setSource(e.target.value)}
                  className="appearance-none bg-gray-800/80 border border-gray-600/50 text-white rounded-lg px-4 py-3 w-full pr-10 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 font-mono tracking-wider"
                >
                  <option value="zoom" className="bg-gray-800">ZOOM CONFERENCE</option>
                  <option value="google_meet" className="bg-gray-800">GOOGLE MEET</option>
                  <option value="custom" className="bg-gray-800">CUSTOM SOFTWARE</option>
                </select>
                <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
                  <svg className="h-5 w-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </div>
            </div>

            {/* Webcam feed with sci-fi overlay */}
            <div className="relative mb-6 rounded-lg overflow-hidden border border-gray-700/50">
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
                className="w-full h-auto block"
              />
              <div className="absolute inset-0 pointer-events-none">
                <div className="absolute inset-0 border-2 border-blue-400/20 rounded-lg"></div>
                <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-blue-400/50 to-transparent"></div>
                <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-blue-400/50 to-transparent"></div>
                {scanning && (
                  <div className="absolute inset-0 bg-blue-900/10 flex items-center justify-center">
                    <div className="animate-pulse text-blue-400 font-mono text-lg tracking-widest">ANALYZING FRAME...</div>
                  </div>
                )}
              </div>
            </div>

            {/* Scan button */}
            <button
              onClick={handleScan}
              disabled={scanning}
              className={`w-full py-3 rounded-lg font-bold transition-all duration-300 relative overflow-hidden group
                ${scanning
                  ? 'bg-gray-800 cursor-not-allowed text-gray-400'
                  : 'bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white shadow-lg hover:shadow-blue-500/30'}`}
            >
              <span className="relative z-10 flex items-center justify-center space-x-2">
                {scanning ? (
                  <>
                    <svg
                      className="animate-spin h-5 w-5 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                      />
                    </svg>
                    <span>SCANNING...</span>
                  </>
                ) : (
                  <>
                    <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v1m6 11h1m-6 0h-1m4-6v1m-8-1v1m-6 4h1m2-8H6m6 4h8m-8 0H6" />
                    </svg>
                    <span>INITIATE SCAN</span>
                  </>
                )}
              </span>
              {!scanning && (
                <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></span>
              )}
            </button>
          </div>
        </div>

        {/* Results panel */}
        {result && (
          <div className={`mt-8 bg-gray-900/80 backdrop-blur-sm border rounded-xl shadow-2xl overflow-hidden transition-all duration-500 ${result.isFake ? 'border-red-500/30' : 'border-green-500/30'}`}>
            <div className="bg-gradient-to-r from-gray-800 to-gray-900 border-b border-gray-700/50 p-4">
              <div className="font-mono text-sm tracking-wider flex items-center">
                <svg className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                ANALYSIS_RESULT
              </div>
            </div>
            <div className="p-6">
              <div className={`flex flex-col items-center justify-center p-6 rounded-lg ${result.isFake ? 'bg-red-900/20' : 'bg-green-900/20'}`}>
                {result.isFake ? (
                  <>
                    <div className="text-red-400 text-4xl mb-4 animate-pulse">
                      <svg className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                    </div>
                    <div className="text-center">
                      <h3 className="text-2xl font-bold text-red-400 mb-2">UNAUTHORIZED ACTIVITY DETECTED</h3>
                      <p className="font-mono text-red-300 tracking-wider">CONFIDENCE LEVEL: <span className="font-bold">{result.confidence.toFixed(2)}%</span></p>
                      <p className="text-red-200 mt-4">Potential cheating attempt identified</p>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="text-green-400 text-4xl mb-4">
                      <svg className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                    <div className="text-center">
                      <h3 className="text-2xl font-bold text-green-400 mb-2">CLEAR VERIFICATION</h3>
                      <p className="font-mono text-green-300 tracking-wider">CONFIDENCE LEVEL: <span className="font-bold">{result.confidence.toFixed(2)}%</span></p>
                      <p className="text-green-200 mt-4">No suspicious activity detected</p>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ScanPage;
