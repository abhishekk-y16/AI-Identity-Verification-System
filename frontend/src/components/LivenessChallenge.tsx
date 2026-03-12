import { useState, useRef, useCallback, useEffect } from 'react'
import Webcam from 'react-webcam'
import { Eye, RotateCcw, CheckCircle, Loader2 } from 'lucide-react'

interface LivenessChallengeProps {
  onComplete: (videoFile: File) => void
  challenges?: string[]
  label?: string
}

const DEFAULT_CHALLENGES = ['Please blink 3 times', 'Turn your head slowly left', 'Turn your head slowly right']

export default function LivenessChallenge({ onComplete, challenges = DEFAULT_CHALLENGES, label = 'Liveness Check' }: LivenessChallengeProps) {
  const webcamRef = useRef<Webcam>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const [currentChallenge, setCurrentChallenge] = useState(0)
  const [recording, setRecording] = useState(false)
  const [completed, setCompleted] = useState(false)
  const [countdown, setCountdown] = useState<number | null>(null)

  const startRecording = useCallback(() => {
    const stream = webcamRef.current?.video?.srcObject as MediaStream
    if (!stream) return

    chunksRef.current = []
    const mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' })
    mediaRecorderRef.current = mediaRecorder

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data)
    }

    mediaRecorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: 'video/webm' })
      const file = new File([blob], 'liveness_video.webm', { type: 'video/webm' })
      onComplete(file)
      setCompleted(true)
    }

    mediaRecorder.start()
    setRecording(true)
    setCurrentChallenge(0)
  }, [onComplete])

  // Advance through challenges with timer
  useEffect(() => {
    if (!recording) return

    const timer = window.setInterval(() => {
      setCurrentChallenge((prev) => {
        if (prev >= challenges.length - 1) {
          // Done with all challenges
          if (mediaRecorderRef.current?.state === 'recording') {
            mediaRecorderRef.current.stop()
          }
          setRecording(false)
          clearInterval(timer)
          return prev
        }
        return prev + 1
      })
    }, 4000)

    return () => clearInterval(timer)
  }, [recording, challenges.length])

  const begin = () => {
    setCountdown(3)
    let c = 3
    const interval = window.setInterval(() => {
      c -= 1
      setCountdown(c)
      if (c <= 0) {
        clearInterval(interval)
        setCountdown(null)
        startRecording()
      }
    }, 1000)
  }

  const reset = () => {
    setCompleted(false)
    setCurrentChallenge(0)
    setRecording(false)
  }

  return (
    <div className="space-y-4">
      <label className="block text-sm font-medium text-gray-700">{label}</label>

      <div className="relative bg-black rounded-xl overflow-hidden">
        {!completed && (
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            videoConstraints={{ facingMode: 'user', width: 640, height: 480 }}
            className="w-full"
          />
        )}

        {completed && (
          <div className="flex flex-col items-center justify-center py-16 bg-green-50">
            <CheckCircle className="h-16 w-16 text-green-500 mb-3" />
            <p className="text-green-700 font-medium">Liveness video captured</p>
          </div>
        )}

        {/* Countdown overlay */}
        {countdown !== null && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50">
            <span className="text-6xl font-bold text-white">{countdown}</span>
          </div>
        )}

        {/* Challenge prompt overlay */}
        {recording && (
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
            <div className="flex items-center gap-2 text-white">
              <Loader2 className="h-5 w-5 animate-spin text-red-400" />
              <span className="text-sm font-medium">
                Challenge {currentChallenge + 1}/{challenges.length}:
              </span>
            </div>
            <p className="text-white text-lg font-semibold mt-1 animate-pulse">
              {challenges[currentChallenge]}
            </p>
          </div>
        )}
      </div>

      <div className="flex justify-center gap-3">
        {completed ? (
          <button onClick={reset} className="btn-secondary flex items-center gap-2">
            <RotateCcw className="h-4 w-4" /> Redo
          </button>
        ) : !recording && countdown === null ? (
          <button onClick={begin} className="btn-primary flex items-center gap-2">
            <Eye className="h-4 w-4" /> Start Liveness Check
          </button>
        ) : null}
      </div>
    </div>
  )
}
