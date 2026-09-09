import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Play, Pause, SkipBack, SkipForward, Volume2 } from "lucide-react";

interface AudioPlayerProps {
  isPlaying: boolean;
  onPlayPause: () => void;
  currentTime?: number;
  duration?: number;
  onSeek?: (time: number) => void;
  onVolumeChange?: (volume: number) => void;
}

export const AudioPlayer = ({ 
  isPlaying, 
  onPlayPause, 
  currentTime = 0, 
  duration = 0,
  onSeek,
  onVolumeChange 
}: AudioPlayerProps) => {
  const [volume, setVolume] = useState([70]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  /**
   * Skip-back and skip-forward previously had NO onClick at all: two permanently
   * inert buttons in a transport bar, which SRS §3.9.1 requires to provide
   * "standard media playback controls". The onSeek prop they need was already
   * supplied by DatapointEditorPanel.
   */
  const SKIP_SECONDS = 5;

  const skipBy = (delta: number) => {
    if (!onSeek) return;
    const target = Math.min(Math.max(currentTime + delta, 0), duration || 0);
    onSeek(target);
  };

  const handleVolumeChange = (newVolume: number[]) => {
    setVolume(newVolume);
    if (onVolumeChange) {
      onVolumeChange(newVolume[0] / 100);
    }
  };

  return (
    <div className="space-y-3">
      {/* Progress bar */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(duration)}</span>
        </div>
        <Slider
          value={[currentTime]}
          onValueChange={(value) => onSeek && onSeek(value[0])}
          max={duration || 100}
          step={0.1}
          className="w-full"
          aria-label="Seek"
        />
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          <Button
            size="sm"
            variant="ghost"
            className="h-8 w-8 p-0"
            aria-label="Skip back 5 seconds"
            disabled={!onSeek}
            onClick={() => skipBy(-SKIP_SECONDS)}
          >
            <SkipBack className="h-4 w-4" aria-hidden="true" />
          </Button>
          
          <Button
            size="sm"
            onClick={onPlayPause}
            className="h-8 w-8 p-0"
            aria-label={isPlaying ? "Pause" : "Play"}
          >
            {isPlaying ? (
              <Pause className="h-4 w-4" aria-hidden="true" />
            ) : (
              <Play className="h-4 w-4" aria-hidden="true" />
            )}
          </Button>
          
          <Button
            size="sm"
            variant="ghost"
            className="h-8 w-8 p-0"
            aria-label="Skip forward 5 seconds"
            disabled={!onSeek}
            onClick={() => skipBy(SKIP_SECONDS)}
          >
            <SkipForward className="h-4 w-4" aria-hidden="true" />
          </Button>
        </div>

        {/* Volume */}
        <div className="flex items-center gap-2">
          <Volume2 className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
          <Slider
            value={volume}
            aria-label="Volume"
            onValueChange={handleVolumeChange}
            max={100}
            step={1}
            className="w-16"
          />
        </div>
      </div>
    </div>
  );
};