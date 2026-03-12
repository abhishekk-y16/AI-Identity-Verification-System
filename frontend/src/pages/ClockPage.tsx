import { useState, useRef, useCallback, useEffect } from 'react'
import { useAuth } from '../hooks/useAuth'
import { attendanceAPI } from '../services/api'
import { Camera, Mic, MicOff, CheckCircle, XCircle, Loader2, Clock } from 'lucide-react'

export default function ClockPage() {
  const { user } = useAuth()
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])

  const [step, setStep] = useState<'ready' | 'capturing' | 'recording' | 'submitting' | 'done'>('ready')
  const [mode, setMode] = useState<'in' | 'out'>('in')
  const [faceImage, setFaceImage] = useState<string>('')
  const [voiceAudio, setVoiceAudio] = useState<string>('')
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState('')
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [todayRecords, setTodayRecords] = useState<any[]>([])

  useEffect(() => {
    attendanceAPI.getToday().then(res => setTodayRecords(res.data)).catch(() => {})
  }, [])

  const startCamera = useCallback(async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: true, audio: true })
      setStream(s)
      if (videoRef.current) {
        videoRef.current.srcObject = s
      }
      setStep('capturing')
      setError('')
      setResult(null)
    } catch {
      setError('Failed to access Camera/Microphone. Please allow permissions.')
    }
  }, [])

  const startVoiceRecording = useCallback(() => {
    if (!stream) return
    setStep('recording')
    audioChunksRef.current = []
    const recorder = new MediaRecorder(stream, { mimeType: 'audio/Webm' })
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunksRef.current.push(e.data)
    }
    recorder.onstop = async () => {
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
      const reader = new FileReader()
      reader.onload = () => {
        const b64 = (reader.result as string).split(',')[1]
        setVoiceAudio(b64)
      }
      reader.readAsDataURL(audioBlob)
    }
    mediaRecorderRef.current = recorder
    recorder.start()
    // Auto-stop after 3 seconds
    setTimeout(() => {
      if (recorder.state === 'recording') {
        recorder.stop()
      }
    }, 3000)
  }, [stream])

  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return
    const canvas = canvasRef.current
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.drawImage(videoRef.current, 0, 0)
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8)
    const b64 = dataUrl.split(',')[1]
    setFaceImage(b64)
    // Start voice recording
    startVoiceRecording()
  }, [startVoiceRecording])

  const stopStream = useCallback(() => {
    stream?.getTracks().forEach(t => t.stop())
    setStream(null)
  }, [stream])

  // Submit once we have both face and voice
  useEffect(() => {
    if (!faceImage || !voiceAudio || step !== 'recording') return
    let cancelled = false
    const submit = async () => {
      setStep('submitting')
      try {
        const payload = { face_image: faceImage, voice_audio: voiceAudio }
        const res = mode === 'in'
          ? await attendanceAPI.clockIn(payload)
          : await attendanceAPI.clockOut(payload)
        if (cancelled) return
        setResult(res.data)
        setStep('done')
        attendanceAPI.getToday().then(r => setTodayRecords(r.data)).catch(() => {})
      } catch (err: any) {
        if (cancelled) return
        setError(err.response?.data?.detail || 'Clock operation failed')
        setStep('ready')
      } finally {
        stopStream()
      }
    }
    submit()
    return () => { cancelled = true }
  }, [faceImage, voiceAudio, step, mode, stopStream])

  const reset = () => {
    setStep('ready')
    setFaceImage('')
    setVoiceAudio('')
    setResult(null)
    setError('')
    stopStream()
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Clock {mode === 'in' ? 'In' : 'Out'}</h1>
        <p className="text-gray-500 mt-1">Verify your identity using face + voice biometrics</p>
      </div>

      {/* Mode toggle */}
      <div className="flex gap-2">
        <button
          onClick={() => { setMode('in'); reset() }}
          className={`flex-1 py-3 rounded-lg font-medium transition-colors ${
            mode === 'in' ? 'bg-green-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          Clock In
        </button>
        <button
          onClick={() => { setMode('out'); reset() }}
          className={`flex-1 py-3 rounded-lg font-medium transition-colors ${
            mode === 'out' ? 'bg-red-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          Clock Out
        </button>
      </div>

      {/* Camera view */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="relative aspect-video bg-gray-900">
          <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
          <canvas ref={canvasRef} className="hidden" />
          {step === 'recording' && (
            <div className="absolute top-4 right-4 flex items-center gap-2 bg-red-600 text-white px-3 py-1 rounded-full text-sm animate-pulse">
              <Mic className="h-4 w-4" /> Recording voice.....
            </div>
          )}
          {step === 'submitting' && (
            <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
              <Loader2 className="h-10 w-10 text-white animate-spin" />
            </div>
          )}
        </div>

        <div className="p-4 space-y-3">
          {step === 'ready' && (
            <button onClick={startCamera} className="w-full py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 flex items-center justify-center gap-2">
              <Camera className="h-5 w-5" />
              Start Verification
            </button>
          )}

          {step === 'capturing' && (
            <button onClick={capturePhoto} className="w-full py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 flex items-center justify-center gap-2">
              <Camera className="h-5 w-5" />
              Capture Face & Record Voice
            </button>
          )}

          {step === 'done' && result && (
            <div className={`p-4 rounded-lg ${result.success ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'}`}>
              <div className="flex items-center gap-3">
                {result.success ? <CheckCircle className="h-6 w-6 text-green-600" /> : <XCircle className="h-6 w-6 text-red-600" />}
                <div>
                  <p className="font-medium">{result.message}</p>
                  {result.face_score !== undefined && (
                    <p className="text-sm text-gray-500 mt-1">
                      Face: {(result.face_score * 100).toFixed(0)}% · Voice: {(result.voice_score * 100).toFixed(0)}%
                    </p>
                  )}
                </div>
              </div>
              <button onClick={reset} className="mt-3 w-full py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 text-sm">
                Done
              </button>
            </div>
          )}

          {error && (
            <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">{error}</div>
          )}
        </div>
      </div>

      {/* Today's records */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-4">
        <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <Clock className="h-5 w-5 text-blue-600" />
          Today's Punches
        </h3>
        {todayRecords.length === 0 ? (
          <p className="text-gray-500 text-sm">No Punches recorded today</p>
        ) : (
          <div className="space-y-2">
            {todayRecords.map((r: any) => (
              <div key={r.record_id} className="flex items-center justify-between py-2 border-b last:border-0">
                <div className="flex items-center gap-3">
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                    r.punch_type === 'clock_in' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                  }`}>
                    {r.punch_type === 'clock_in' ? 'IN' : 'OUT'}
                  </span>
                  <span className="text-sm text-gray-900">
                    {new Date(r.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <span className={`text-xs px-2 py-0.5 rounded ${
                  r.status === 'on_time' ? 'bg-green-50 text-green-600' :
                  r.status === 'late' ? 'bg-yellow-50 text-yellow-600' :
                  'bg-gray-50 text-gray-600'
                }`}>
                  {r.status?.replace('_', ' ')}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
