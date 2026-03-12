import { useState, useRef, useCallback } from 'react'
import { Mic, Square, RotateCcw, Check } from 'lucide-react'

interface AudioCaptureProps {
  onCapture: (file: File) => void
  duration?: number
  label?: string
}

export default function AudioCapture({ onCapture, duration = 5, label = 'Record Voice' }: AudioCaptureProps) {
  const [recording, setRecording] = useState(false)
  const [recorded, setRecorded] = useState(false)
  const [timeLeft, setTimeLeft] = useState(duration)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<number | null>(null)

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const file = new File([blob], 'voice_capture.webm', { type: 'audio/webm' })
        onCapture(file)
        setRecorded(true)
        stream.getTracks().forEach((track) => track.stop())
      }

      mediaRecorder.start()
      setRecording(true)
      setTimeLeft(duration)

      let remaining = duration
      timerRef.current = window.setInterval(() => {
        remaining -= 1
        setTimeLeft(remaining)
        if (remaining <= 0) {
          mediaRecorder.stop()
          setRecording(false)
          if (timerRef.current) clearInterval(timerRef.current)
        }
      }, 1000)
    } catch {
      alert('Microphone access denied. Please allow microphone permissions.')
    }
  }, [duration, onCapture])

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop()
      setRecording(false)
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }

  const reset = () => {
    setRecorded(false)
    setTimeLeft(duration)
  }

  return (
    <div className="space-y-4">
      <label className="block text-sm font-medium text-gray-700">{label}</label>

      <div className="flex flex-col items-center gap-4 p-6 bg-gray-50 rounded-xl">
        {/* Visualizer circle */}
        <div className={`w-24 h-24 rounded-full flex items-center justify-center transition-all ${
          recording ? 'bg-red-100 animate-pulse-slow' : recorded ? 'bg-green-100' : 'bg-gray-200'
        }`}>
          {recording ? (
            <div className="text-2xl font-bold text-red-600">{timeLeft}s</div>
          ) : recorded ? (
            <Check className="h-10 w-10 text-green-600" />
          ) : (
            <Mic className="h-10 w-10 text-gray-400" />
          )}
        </div>

        {recording && (
          <p className="text-sm text-gray-500">Please say: "My voice is my identity"</p>
        )}

        <div className="flex gap-3">
          {recorded ? (
            <button onClick={reset} className="btn-secondary flex items-center gap-2">
              <RotateCcw className="h-4 w-4" /> Re-record
            </button>
          ) : recording ? (
            <button onClick={stopRecording} className="btn-danger flex items-center gap-2">
              <Square className="h-4 w-4" /> Stop
            </button>
          ) : (
            <button onClick={startRecording} className="btn-primary flex items-center gap-2">
              <Mic className="h-4 w-4" /> {label} ({duration}s)
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
