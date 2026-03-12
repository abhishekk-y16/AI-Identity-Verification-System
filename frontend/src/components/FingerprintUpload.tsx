import { useState, useRef } from 'react'
import { Upload, RotateCcw, Check, Fingerprint } from 'lucide-react'

interface FingerprintUploadProps {
  onCapture: (file: File) => void
  label?: string
}

export default function FingerprintUpload({ onCapture, label = 'Upload Fingerprint' }: FingerprintUploadProps) {
  const [preview, setPreview] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file.')
      return
    }
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target?.result as string)
    reader.readAsDataURL(file)
    onCapture(file)
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }

  const reset = () => {
    setPreview(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="space-y-4">
      <label className="block text-sm font-medium text-gray-700">{label}</label>

      {preview ? (
        <div className="flex flex-col items-center gap-4 p-4 bg-gray-50 rounded-xl">
          <div className="relative w-48 h-48 rounded-lg overflow-hidden bg-black">
            <img src={preview} alt="Fingerprint" className="w-full h-full object-contain" />
            <div className="absolute top-2 right-2 bg-green-500 text-white rounded-full p-1">
              <Check className="h-4 w-4" />
            </div>
          </div>
          <button onClick={reset} className="btn-secondary flex items-center gap-2">
            <RotateCcw className="h-4 w-4" /> Replace
          </button>
        </div>
      ) : (
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => inputRef.current?.click()}
          className="flex flex-col items-center gap-3 p-8 border-2 border-dashed border-gray-300 rounded-xl cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-colors"
        >
          <Fingerprint className="h-12 w-12 text-gray-400" />
          <div className="text-center">
            <p className="text-sm font-medium text-gray-700">Drag & drop or click to upload</p>
            <p className="text-xs text-gray-500 mt-1">PNG, JPG, BMP — max 10MB</p>
          </div>
          <Upload className="h-5 w-5 text-gray-400" />
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={handleChange}
        className="hidden"
      />
    </div>
  )
}
