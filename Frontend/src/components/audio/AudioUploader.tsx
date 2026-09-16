import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Card, CardContent } from "@/components/ui/card";
import { Upload, FileAudio } from "lucide-react";
import { toast } from "sonner";
import { API_BASE, AudioReference } from '@/lib/api';
import { isAcceptedAudioFile } from '@/lib/audioFiles';
import { mapWithConcurrency } from '@/lib/concurrency';
import { describeHttpError } from '@/lib/httpError';

/** Simultaneous uploads from one drop; uploads are bandwidth-bound, so more buys nothing. */
const UPLOAD_CONCURRENCY = 2;

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
        throw await describeHttpError(response);
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

  const onDrop = useCallback((acceptedFiles: File[], fileRejections: { file: File }[] = []) => {
    fileRejections.forEach(({ file }) => {
      toast.error(`Invalid file type: ${file.name}. Supported formats: WAV, MP3, M4A, FLAC`);
    });
    // Bounded, not `forEach(async …)`: dropping a folder of 50 recordings used
    // to open 50 simultaneous uploads of up to 100 MB each.
    void mapWithConcurrency(acceptedFiles, UPLOAD_CONCURRENCY, async (file) => {
      if (isAcceptedAudioFile(file)) {
        try {
          await uploadFile(file);
        } catch {
          // Error already handled in uploadFile; one failure must not stop the rest.
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
                <p className="text-xs text-muted-foreground mt-1">
                  Up to 100 MB and 10 minutes per file
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  );
};
