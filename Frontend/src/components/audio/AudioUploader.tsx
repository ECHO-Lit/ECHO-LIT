import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Card, CardContent } from "@/components/ui/card";
import { Upload, FileAudio } from "lucide-react";
import { toast } from "sonner";
import { API_BASE, AudioReference } from '@/lib/api';

interface AudioUploaderProps {
  onUploadSuccess?: (uploadResponse: AudioReference) => void;
  model?: string;
}

export const AudioUploader = ({ onUploadSuccess, model }: AudioUploaderProps) => {
  const uploadFile = async (file: File) => {
    
    const formData = new FormData();
    formData.append('file', file);
    // Kept during the API compatibility window; upload no longer runs inference.
    formData.append('model', model || 'whisper-base');

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        credentials: 'include',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await response.json();
      toast.success(`Uploaded: ${file.name}`);
      
      // Call the callback with upload response
      if (onUploadSuccess) {
        onUploadSuccess(data);
      }
      
      return data;
    } catch (error) {
      console.error('Upload error:', error);
      toast.error(`Failed to upload ${file.name}: ${error instanceof Error ? error.message : 'Unknown error'}`);
      throw error;
    }
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    acceptedFiles.forEach(async (file) => {
      
      // Check both MIME type and file extension for better .flac support
      const allowedExtensions = ['.wav', '.mp3', '.m4a', '.flac'];
      const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
      const isValidFile = file.type.startsWith('audio/') || allowedExtensions.includes(fileExtension);
      
      if (isValidFile) {
        try {
          await uploadFile(file);
        } catch {
          // Error already handled in uploadFile
        }
      } else {
        console.warn('Invalid file type:', file.type);
        toast.error(`Invalid file type: ${file.name}. Supported formats: WAV, MP3, M4A, FLAC`);
      }
    });
  }, [onUploadSuccess, model]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.wav', '.mp3', '.m4a', '.flac']
    },
    multiple: true
  });

  return (
    <>
      {/* Upload drop zone overlay - only visible when dragging */}
      <div
        {...getRootProps()}
        className={`
          fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center
          transition-opacity duration-200
          ${isDragActive ? 'opacity-100' : 'opacity-0 pointer-events-none'}
        `}
      >
        <input {...getInputProps()} />
        <Card className="w-96 border-2 border-dashed border-primary">
          <CardContent className="p-8 text-center">
            <div className="flex flex-col items-center gap-4">
              <div className="p-4 rounded-full bg-primary/10">
                <FileAudio className="h-8 w-8 text-primary" />
              </div>
              <div>
                <h3 className="font-medium">Drop files here</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Supports WAV, MP3, M4A, FLAC
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  );
};
