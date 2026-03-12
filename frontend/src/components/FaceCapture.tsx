import { useRef, useCallback, useState } from 'react'
import Webcam from 'react-webcam'
import { Camera, RotateCcw, Check } from 'lucide-react'

interface FaceCaptureProps {
  onCapture: (file: File) => void
  label?: string
}

export default function FaceCapture({ onCapture, label = 'Capture Face' }: FaceCaptureProps) {
  const webcamRef = useRef<Webcam>(null)
  const [captured, setCaptured] = useState<string | null>(null)

  const capture = useCallback(() => {
    if (!webcamRef.current) return
    const imageSrc = webcamRef.current.getScreenshot()
    if (imageSrc) {
      setCaptured(imageSrc)
      // Convert base64 to File
      fetch(imageSrc)
        .then((res) => res.blob())
        .then((blob) => {
          const file = new File([blob], 'face_capture.jpg', { type: 'image/jpeg' })
          onCapture(file)
        })
    }
  }, [onCapture])

  const reset = () => setCaptured(null)

  return (
    <div className="space-y-4">
      <label className="block text-sm font-medium text-gray-700">{label}</label>
      <div className="relative rounded-xl overflow-hidden bg-black aspect-[4/3] max-w-md mx-auto">
        {captured ? (
          <img src={captured} alt="Captured face" className="w-full h-full object-cover" />
        ) : (
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            videoConstraints={{ facingMode: 'user', width: 640, height: 480 }}
            className="w-full h-full object-cover"
          />
        )}
        {/* Face guide overlay */}
        {!captured && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="w-48 h-60 border-2 border-white/50 rounded-full" />
          </div>
        )}
      </div>
      <div className="flex justify-center gap-3">
        {captured ? (
          <>
            <button onClick={reset} className="btn-secondary flex items-center gap-2">
              <RotateCcw className="h-4 w-4" /> Retake
            </button>
            <div className="flex items-center gap-2 text-green-600 font-medium">
              <Check className="h-5 w-5" /> Captured
            </div>
          </>
        ) : (
          <button onClick={capture} className="btn-primary flex items-center gap-2">
            <Camera className="h-4 w-4" /> {label}
          </button>
        )}
      </div>
    </div>
  )
}
