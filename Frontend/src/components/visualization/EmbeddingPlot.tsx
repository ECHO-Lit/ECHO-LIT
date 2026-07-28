import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import Plot from "react-plotly.js";
import { useEmbedding } from "../../contexts/EmbeddingContext";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ZoomIn, ZoomOut, RotateCcw, Layers3, Target } from "lucide-react";
import { clusterColor, clusterName, clusterSymbol, NOISE_LABEL } from "@/lib/clusterPalette";

export type EmbeddingColorMode = 'default' | 'cluster';

interface EmbeddingPlotProps {
  selectedMethod?: string;
  is3D?: boolean;
  onPointSelect?: (filename: string, coordinates: number[]) => void;
  onAngleRangeSelect?: (selectedFiles: string[]) => void;
  selectedFile?: string | null;
  selectionMode?: 'box' | 'lasso';
  onSelectionChange?: (selectedFiles: string[]) => void;
  colorMode?: EmbeddingColorMode;
}

type PlaneType = 'none' | 'xy' | 'xz' | 'yz';

// Separate component for the actual plot content
interface EmbeddingPlotContentProps {
  selectedMethod: string;
  is3D: boolean;
  onPointSelect?: (filename: string, coordinates: number[]) => void;
  onAngleRangeSelect?: (selectedFiles: string[]) => void;
  selectedFile?: string | null;
  selectionMode?: 'box' | 'lasso';
  onSelectionChange?: (selectedFiles: string[]) => void;
  colorMode?: EmbeddingColorMode;
}

const EmbeddingPlotContent = ({ selectedMethod, is3D, onPointSelect, onAngleRangeSelect, selectedFile, selectionMode = 'box', onSelectionChange, colorMode = 'default' }: EmbeddingPlotContentProps) => {
  const { embeddingData, isLoading, error } = useEmbedding();
  const plotRef = useRef<any>(null);
  const [selectedPlane, setSelectedPlane] = useState<PlaneType>('none');
  const [angleMin, setAngleMin] = useState<number>(40);
  const [angleMax, setAngleMax] = useState<number>(50);
  const [selectedByAngle, setSelectedByAngle] = useState<string[]>([]);

  // Reset plane selection when switching to 2D
  useEffect(() => {
    if (!is3D) {
      setSelectedPlane('none');
      setSelectedByAngle([]);
    }
  }, [is3D]);

  // Generate mock data as fallback
  const generateMockData = () => {
    const n = 50;
    const x = [];
    const y = [];
    const colors = [];
    const text = [];
    
    for (let i = 0; i < n; i++) {
      x.push(Math.random() * 20 - 10);
      y.push(Math.random() * 20 - 10);
      colors.push(['neutral', 'happy', 'sad', 'angry'][Math.floor(Math.random() * 4)]);
      text.push(`Sample ${i + 1}`);
    }
    
    return { x, y, colors, text };
  };

  // Handle point selection
  const handlePointClick = useCallback((event: any) => {
    if (event.points && event.points.length > 0) {
      const point = event.points[0];
      // Use customdata[0] which contains the raw filename (not the HTML-formatted text)
      const filename = point.customdata[0];
      const coordinates = is3D ? [point.x, point.y, point.z] : [point.x, point.y];
      
      if (onPointSelect) {
        onPointSelect(filename, coordinates);
      }
    }
  }, [onPointSelect, is3D]);

  // Handle 2D box/lasso selection
  const handleSelection = useCallback((event: any) => {
    if (!is3D && onSelectionChange && event?.points) {
      // Use customdata[0] which contains the raw filename (not the HTML-formatted text)
      const selected = event.points.map((p: any) => p.customdata[0]);
      onSelectionChange(selected);
    }
  }, [is3D, onSelectionChange]);

  // Handle deselection
  const handleDeselect = useCallback(() => {
    if (!is3D && onSelectionChange) {
      onSelectionChange([]);
    }
  }, [is3D, onSelectionChange]);

  // Use real embedding data if available, otherwise fall back to mock data
  const getPlotData = () => {
    if (embeddingData && embeddingData.reduced_embeddings && embeddingData.reduced_embeddings.length > 0) {
      const x = embeddingData.reduced_embeddings.map(point => point.coordinates[0]);
      const y = embeddingData.reduced_embeddings.map(point => point.coordinates[1]);
      const z = is3D && embeddingData.reduced_embeddings[0].coordinates.length > 2 
        ? embeddingData.reduced_embeddings.map(point => point.coordinates[2]) 
        : undefined;
      const text = embeddingData.reduced_embeddings.map(point => point.filename);
      
      // Enhanced color mapping with spatial clustering
      const colors = embeddingData.reduced_embeddings.map((point, index) => {
        const filename = point.filename.toLowerCase();
        
        // First try emotion-based coloring from RAVDESS dataset
        if (filename.includes('01-01') || filename.includes('neutral')) return 'neutral';
        if (filename.includes('01-03') || filename.includes('happy') || filename.includes('joy')) return 'happy';
        if (filename.includes('01-04') || filename.includes('sad') || filename.includes('sadness')) return 'sad';
        if (filename.includes('01-05') || filename.includes('angry') || filename.includes('anger')) return 'angry';
        if (filename.includes('01-06') || filename.includes('fear') || filename.includes('afraid')) return 'fear';
        if (filename.includes('01-07') || filename.includes('disgust')) return 'disgust';
        if (filename.includes('01-08') || filename.includes('surprise')) return 'surprise';
        if (filename.includes('01-02') || filename.includes('calm')) return 'calm';
        
        // For Common Voice or other datasets, use spatial clustering
        const coords = point.coordinates;
        if (coords.length >= 2) {
          const [px, py] = coords;
          
          // Calculate quartiles for better spatial distribution
          const sortedX = x.slice().sort((a, b) => a - b);
          const sortedY = y.slice().sort((a, b) => a - b);
          const q1X = sortedX[Math.floor(sortedX.length * 0.25)];
          const q3X = sortedX[Math.floor(sortedX.length * 0.75)];
          const q1Y = sortedY[Math.floor(sortedY.length * 0.25)];
          const q3Y = sortedY[Math.floor(sortedY.length * 0.75)];
          
          // Assign colors based on spatial regions
          if (px > q3X && py > q3Y) return 'region1'; // Top-right
          if (px < q1X && py > q3Y) return 'region2'; // Top-left
          if (px < q1X && py < q1Y) return 'region3'; // Bottom-left
          if (px > q3X && py < q1Y) return 'region4'; // Bottom-right
          if (px >= q1X && px <= q3X && py >= q1Y && py <= q3Y) return 'center'; // Center
          if (px >= q1X && px <= q3X) return 'mid_vertical'; // Middle band
          if (py >= q1Y && py <= q3Y) return 'mid_horizontal'; // Middle band
        }
        
        return 'unknown';
      });
      
      return { x, y, z, colors, text };
    }
    
    const mockData = generateMockData();
    if (is3D) {
      // Generate mock Z coordinates
      const z = mockData.x.map(() => Math.random() * 20 - 10);
      return { ...mockData, z };
    }
    return mockData;
  };

  // Create transparent plane surfaces for 3D visualization
  const createPlane = (planeType: PlaneType, bounds: { x: [number, number], y: [number, number], z: [number, number] }) => {
    if (!is3D || planeType === 'none') return null;

    // Make bounds bigger for more visible plane
    const [xMin, xMax] = [bounds.x[0] * 1.3, bounds.x[1] * 1.3];
    const [yMin, yMax] = [bounds.y[0] * 1.3, bounds.y[1] * 1.3];
    const [zMin, zMax] = [bounds.z[0] * 1.3, bounds.z[1] * 1.3];

    const planeAlpha = 0.35; // Increased opacity (35% instead of 20%)
    
    switch (planeType) {
      case 'xy': // X-Y plane through origin (Z = 0)
        return {
          type: 'surface' as const,
          x: [[xMin, xMax], [xMin, xMax]],
          y: [[yMin, yMin], [yMax, yMax]],
          z: [[0, 0], [0, 0]], // Always pass through Z = 0 (origin)
          opacity: planeAlpha,
          colorscale: [[0, 'rgba(59, 130, 246, 0.5)'], [1, 'rgba(59, 130, 246, 0.5)']], // Blue with higher opacity
          showscale: false,
          hoverinfo: 'skip',
          name: 'X-Y Plane (Z=0)'
        };
      case 'xz': // X-Z plane through origin (Y = 0)
        return {
          type: 'surface' as const,
          x: [[xMin, xMax], [xMin, xMax]],
          y: [[0, 0], [0, 0]], // Always pass through Y = 0 (origin)
          z: [[zMin, zMin], [zMax, zMax]],
          opacity: planeAlpha,
          colorscale: [[0, 'rgba(16, 185, 129, 0.5)'], [1, 'rgba(16, 185, 129, 0.5)']], // Green with higher opacity
          showscale: false,
          hoverinfo: 'skip',
          name: 'X-Z Plane (Y=0)'
        };
      case 'yz': // Y-Z plane through origin (X = 0)
        return {
          type: 'surface' as const,
          x: [[0, 0], [0, 0]], // Always pass through X = 0 (origin)
          y: [[yMin, yMax], [yMin, yMax]],
          z: [[zMin, zMin], [zMax, zMax]],
          opacity: planeAlpha,
          colorscale: [[0, 'rgba(239, 68, 68, 0.5)'], [1, 'rgba(239, 68, 68, 0.5)']], // Red with higher opacity
          showscale: false,
          hoverinfo: 'skip',
          name: 'Y-Z Plane (X=0)'
        };
      default:
        return null;
    }
  };

  // Calculate angle between point and selected plane relative to origin (0,0,0)
  const calculateAngleToPlane = (x: number, y: number, z: number, plane: PlaneType): number => {
    if (plane === 'none') return 0;
    
    const point = [x, y, z];
    const origin = [0, 0, 0];
    
    // Calculate vector from origin to point
    const vector = [x - origin[0], y - origin[1], z - origin[2]];
    const vectorMagnitude = Math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2);
    
    if (vectorMagnitude === 0) return 0; // Point at origin
    
    // Define plane normal vectors
    let planeNormal: number[];
    switch (plane) {
      case 'xy': planeNormal = [0, 0, 1]; break; // Z axis (normal to XY plane)
      case 'xz': planeNormal = [0, 1, 0]; break; // Y axis (normal to XZ plane)  
      case 'yz': planeNormal = [1, 0, 0]; break; // X axis (normal to YZ plane)
      default: planeNormal = [0, 0, 1]; break;
    }
    
    // Calculate dot product
    const dotProduct = vector[0] * planeNormal[0] + vector[1] * planeNormal[1] + vector[2] * planeNormal[2];
    
    // Calculate angle between vector and plane normal (0° = perpendicular to plane, 90° = in plane)
    const angleToNormal = Math.acos(Math.abs(dotProduct) / vectorMagnitude) * (180 / Math.PI);
    
    // Convert to angle from plane (90° - angle to normal)
    return 90 - angleToNormal;
  };

  // Select points based on angle range - memoized to prevent unnecessary recalculations
  const selectedFiles = useMemo(() => {
    if (!is3D || selectedPlane === 'none' || !embeddingData?.reduced_embeddings) {
      return [];
    }

    return embeddingData.reduced_embeddings
      .filter(point => {
        if (point.coordinates.length < 3) return false;
        
        const [x, y, z] = point.coordinates;
        const angle = calculateAngleToPlane(x, y, z, selectedPlane);
        
        return angle >= angleMin && angle <= angleMax;
      })
      .map(point => point.filename);
  }, [is3D, selectedPlane, embeddingData, angleMin, angleMax]);

  // Update selected points when calculated files change
  useEffect(() => {
    setSelectedByAngle(selectedFiles);
    
    // Notify parent component only if selection actually changed
    if (onAngleRangeSelect && selectedFiles.join(',') !== selectedByAngle.join(',')) {
      onAngleRangeSelect(selectedFiles);
    }
  }, [selectedFiles, onAngleRangeSelect]); // Remove selectedByAngle from dependencies to prevent loops

  const plotData = getPlotData();
  const { x, y, colors, text } = plotData;
  const z = 'z' in plotData ? plotData.z : undefined;

  // Calculate bounds for plane creation
  const bounds = x.length > 0 ? {
    x: [Math.min(...x) * 1.1, Math.max(...x) * 1.1] as [number, number],
    y: [Math.min(...y) * 1.1, Math.max(...y) * 1.1] as [number, number],
    z: z && z.length > 0 ? [Math.min(...z) * 1.1, Math.max(...z) * 1.1] as [number, number] : [0, 0] as [number, number]
  } : { x: [0, 0] as [number, number], y: [0, 0] as [number, number], z: [0, 0] as [number, number] };

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-xs text-muted-foreground flex items-center gap-2">
          <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
          Loading embeddings...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center p-4">
        <div className="text-xs text-red-500 text-center">
          <div className="font-medium">⚠️ Error loading embeddings</div>
          <div className="mt-1">{error}</div>
        </div>
      </div>
    );
  }

  // Create marker sizes based on selection
  const markerSizes = text.map(filename => {
    if (selectedFile === filename) return 12; // Currently selected file (medium-large)
    if (selectedByAngle.includes(filename)) return 8; // Angle range selected (medium)
    return 6; // Default (smaller)
  });

  // Create marker colors based on selection
  const markerColors = text.map(filename => {
    if (selectedFile === filename) return '#FFD700'; // Gold for selected file
    if (selectedByAngle.includes(filename)) return '#ef4444'; // Red for angle selected
    return '#3b82f6'; // Blue for all other points
  });

  // Create marker opacities based on selection
  const hasSelection = selectedFile || selectedByAngle.length > 0;
  const markerOpacities = text.map(filename => {
    if (!hasSelection) return 0.8; // Default opacity when no selection
    if (selectedFile === filename) return 1.0; // Full opacity for selected file
    if (selectedByAngle.includes(filename)) return 0.9; // High opacity for angle selected
    // Different transparency for 2D vs 3D unselected points
    return is3D ? 0.1 : 0.45; // More transparent in 3D, slightly visible in 2D
  });

  // Create traces array - start with main scatter plot
  const traces: any[] = [];

  // Create hover text with angle information
  const hoverText = text.map((filename, index) => {
    let baseText = `<b>${filename}</b>`;
    
    // Add angle information if this point is selected by angle range and in 3D mode
    if (is3D && selectedPlane !== 'none' && selectedByAngle.includes(filename) && z) {
      const [px, py, pz] = [x[index], y[index], z[index]];
      const angle = calculateAngleToPlane(px, py, pz, selectedPlane);
      baseText += `<br>Angle: ${angle.toFixed(1)}°`;
      baseText += `<br>Plane: ${selectedPlane.toUpperCase()}`;
    }
    
    return baseText;
  });

  // Create main trace data
  const traceData: any = {
    x: x,
    y: y,
    mode: 'markers',
    type: is3D ? 'scatter3d' : 'scatter',
    marker: {
      size: markerSizes,
      color: markerColors,
      showscale: false,
      line: {
        width: 0, // Remove marker outlines
        color: 'transparent'
      },
      opacity: markerOpacities // Use dynamic opacity array
    },
    text: hoverText,
    customdata: text.map((filename, index) => [filename, colors[index]]), // Store [filename, color] for each point
  };

  // Add Z coordinate for 3D plots
  if (is3D && z) {
    traceData.z = z;
    traceData.hovertemplate = '%{text}<extra></extra>';
  } else {
    traceData.hovertemplate = '%{text}<extra></extra>';
  }

  // Cluster view: one trace per cluster, so Plotly renders a clickable legend and
  // each cluster can be isolated or hidden. Identity is carried by hue AND marker
  // symbol together -- the palette's colour-blindness margin in this size range is
  // only safe with that second channel.
  const clusterMap = embeddingData?.clusterByFilename;
  const showClusters = colorMode === 'cluster' && !!clusterMap;

  if (showClusters) {
    const labelsPresent = Array.from(new Set(text.map((filename) => clusterMap[filename] ?? NOISE_LABEL)));
    // Noise last so it is drawn underneath the real clusters and reads as background.
    labelsPresent.sort((a, b) => (a === NOISE_LABEL ? 1 : b === NOISE_LABEL ? -1 : a - b));

    for (const label of labelsPresent) {
      const indices = text
        .map((filename, index) => ({ filename, index }))
        .filter(({ filename }) => (clusterMap[filename] ?? NOISE_LABEL) === label)
        .map(({ index }) => index);
      if (indices.length === 0) continue;

      const isNoise = label === NOISE_LABEL;
      const clusterTrace: any = {
        x: indices.map((i) => x[i]),
        y: indices.map((i) => y[i]),
        mode: 'markers',
        type: is3D ? 'scatter3d' : 'scatter',
        name: `${clusterName(label)} (${indices.length})`,
        marker: {
          // Keep the selected-file highlight visible on top of cluster colouring.
          size: indices.map((i) => (selectedFile === text[i] ? 12 : isNoise ? 5 : 7)),
          color: indices.map((i) =>
            selectedFile === text[i] ? '#FFD700' : clusterColor(label, false),
          ),
          symbol: clusterSymbol(label),
          opacity: isNoise ? 0.55 : 0.85,
          line: { width: 0, color: 'transparent' },
          showscale: false,
        },
        text: indices.map((i) => `<b>${text[i]}</b><br>${clusterName(label)}`),
        customdata: indices.map((i) => [text[i], String(label)]),
        hovertemplate: '%{text}<extra></extra>',
        showlegend: true,
      };
      if (is3D && z) clusterTrace.z = indices.map((i) => z[i]);
      traces.push(clusterTrace);
    }
  } else {
    traces.push(traceData);
  }

  // Add origin point (0,0,0) highlight for 3D plots or (0,0) for 2D plots
  const originTrace: any = {
    x: [0],
    y: [0],
    mode: 'markers',
    type: is3D ? 'scatter3d' : 'scatter',
    marker: {
      size: is3D ? 5 : 4, // Slightly smaller to match the new scale
      color: '#000000', // Black for origin
      symbol: 'diamond',
      line: {
        width: 1, // Thinner outline
        color: '#ffffff' // White outline for visibility
      },
      opacity: 0.8 // Slightly transparent
    },
    text: [is3D ? 'Origin (0,0,0)' : 'Origin (0,0)'],
    hovertemplate: is3D ? '<b>Origin (0,0,0)</b><extra></extra>' : '<b>Origin (0,0)</b><extra></extra>',
    name: 'Origin',
    showlegend: false
  };

  if (is3D) {
    originTrace.z = [0];
  }

  traces.push(originTrace);

  // Add plane if selected and in 3D mode
  if (is3D && selectedPlane !== 'none') {
    const planeTrace = createPlane(selectedPlane, bounds);
    if (planeTrace) {
      traces.push(planeTrace);
    }
  }

  // Layout configuration
  const layout: any = {
    autosize: true,
    margin: { l: 35, r: 35, t: 35, b: 35 },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white',
    // The cluster legend is the identity key -- without it colour alone would have
    // to carry which group is which.
    showlegend: showClusters,
    legend: {
      font: { size: 9 },
      orientation: 'h',
      yanchor: 'bottom',
      y: -0.18,
      x: 0,
      bgcolor: 'rgba(255,255,255,0.85)',
    },
    font: {
      size: 11,
      color: '#374151'
    },
    dragmode: is3D ? 'orbit' : (selectionMode === 'box' ? 'select' : 'lasso'),
    hovermode: 'closest',
    uirevision: true // Maintains UI state on data updates
  };

  if (is3D) {
    // 3D scene configuration
    layout.scene = {
      xaxis: {
        showgrid: false, // Remove grid lines
        gridcolor: '#e5e7eb',
        showticklabels: false,
        title: { text: 'X', font: { size: 10 } },
        backgroundcolor: 'white',
        showspikes: false,
        zeroline: false, // Remove zero line
        showline: false  // Remove axis line
      },
      yaxis: {
        showgrid: false, // Remove grid lines
        gridcolor: '#e5e7eb',
        showticklabels: false,
        title: { text: 'Y', font: { size: 10 } },
        backgroundcolor: 'white',
        showspikes: false,
        zeroline: false, // Remove zero line
        showline: false  // Remove axis line
      },
      zaxis: {
        showgrid: false, // Remove grid lines
        gridcolor: '#e5e7eb',
        showticklabels: false,
        title: { text: 'Z', font: { size: 10 } },
        backgroundcolor: 'white',
        showspikes: false,
        zeroline: false, // Remove zero line
        showline: false  // Remove axis line
      },
      bgcolor: 'white',
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.5 },
        center: { x: 0, y: 0, z: 0 },
        up: { x: 0, y: 0, z: 1 }
      },
      aspectmode: 'cube',
      dragmode: 'orbit'
    };
  } else {
    // 2D axis configuration with enhanced zoom support
    layout.xaxis = {
      showgrid: true,
      gridcolor: '#e5e7eb',
      showticklabels: false,
      title: { text: 'X', font: { size: 10 } },
      zeroline: true,
      zerolinecolor: '#d1d5db',
      zerolinewidth: 1,
      fixedrange: false // Allow zoom
    };
    layout.yaxis = {
      showgrid: true,
      gridcolor: '#e5e7eb',
      showticklabels: false,
      title: { text: 'Y', font: { size: 10 } },
      zeroline: true,
      zerolinecolor: '#d1d5db',
      zerolinewidth: 1,
      fixedrange: false // Allow zoom
    };
  }

  // Add compact annotation
  if (embeddingData) {
    layout.annotations = [{
      text: `${embeddingData.total_files} files • ${is3D ? '3D' : '2D'}`,
      xref: 'paper',
      yref: 'paper',
      x: 0.02,
      y: 0.98,
      xanchor: 'left',
      yanchor: 'top',
      font: { size: 9, color: '#6b7280' },
      showarrow: false,
      bgcolor: 'rgba(255,255,255,0.8)',
      bordercolor: '#e5e7eb',
      borderwidth: 1,
      borderpad: 2
    }];
  }

  return (
    <div className="w-full h-full min-h-0 relative">
      {/* Plane Selection Controls - Only show in 3D mode */}
      {is3D && (
        <div className="absolute top-2 right-2 z-10 bg-white/95 backdrop-blur-sm border border-gray-200 rounded-md p-2 shadow-sm">
          {/* Plane Selection */}
          <div className="flex items-center gap-2 mb-2">
            <Layers3 className="h-3 w-3 text-gray-600" />
            <span className="text-xs text-gray-600 font-medium">Plane:</span>
            <Select
              value={selectedPlane}
              onValueChange={(value: PlaneType) => setSelectedPlane(value)}
            >
              <SelectTrigger className="w-16 h-6 text-xs border-gray-300 hover:border-gray-400 transition-colors">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None</SelectItem>
                <SelectItem value="xy">X-Y</SelectItem>
                <SelectItem value="xz">X-Z</SelectItem>
                <SelectItem value="yz">Y-Z</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          {/* Angle Range Selector */}
          {selectedPlane !== 'none' && (
            <div className="space-y-2 pt-2 border-t border-gray-200">
              <div className="flex items-center gap-1">
                <Target className="h-3 w-3 text-gray-600" />
                <span className="text-xs text-gray-600 font-medium">Angle Range:</span>
              </div>
              
              <div className="flex items-center gap-1">
                <Input
                  type="number"
                  min="0"
                  max="90"
                  step="1"
                  value={angleMin}
                  onChange={(e) => setAngleMin(Number(e.target.value))}
                  className="w-14 h-6 text-xs text-center px-1"
                />
                <span className="text-xs text-gray-500">-</span>
                <Input
                  type="number"
                  min="0"
                  max="90"
                  step="1"
                  value={angleMax}
                  onChange={(e) => setAngleMax(Number(e.target.value))}
                  className="w-14 h-6 text-xs text-center px-1"
                />
                <span className="text-xs text-gray-500">°</span>
              </div>
              
              {selectedByAngle.length > 0 && (
                <div className="text-[10px] text-red-600 bg-red-50 px-2 py-1 rounded">
                  🔴 {selectedByAngle.length} points selected
                </div>
              )}
            </div>
          )}
          
          {selectedPlane !== 'none' && (
            <div className="text-[10px] text-gray-500 mt-1">
              {selectedPlane === 'xy' && '🔵 Blue plane: X-Y (Z=0)'}
              {selectedPlane === 'xz' && '🟢 Green plane: X-Z (Y=0)'}
              {selectedPlane === 'yz' && '🔴 Red plane: Y-Z (X=0)'}
            </div>
          )}
        </div>
      )}
      
      <Plot
        ref={plotRef}
        data={traces}
        layout={layout}
        onClick={handlePointClick}
        onSelected={handleSelection}
        onDeselect={handleDeselect}
        config={{
          displayModeBar: false, // Hide the mode bar completely
          displaylogo: false,
          responsive: true,
          autosizable: true,
          scrollZoom: true,
          doubleClick: 'reset+autosize',
          showTips: false, // Hide hover tips,
          toImageButtonOptions: {
            format: 'png',
            filename: `embeddings_${selectedMethod}_${is3D ? '3D' : '2D'}${selectedPlane !== 'none' ? `_${selectedPlane}` : ''}`,
            height: 800,
            width: 800,
            scale: 2
          }
        }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
      />
    </div>
  );
};

export const EmbeddingPlot = ({ selectedMethod = "pca", is3D = false, onPointSelect, onAngleRangeSelect, selectedFile, selectionMode = 'box', onSelectionChange, colorMode = 'default' }: EmbeddingPlotProps) => {
  return (
    <div className="w-full h-full min-h-0 relative">
      <EmbeddingPlotContent
        selectedMethod={selectedMethod}
        is3D={is3D}
        onPointSelect={onPointSelect}
        onAngleRangeSelect={onAngleRangeSelect}
        selectedFile={selectedFile}
        selectionMode={selectionMode}
        onSelectionChange={onSelectionChange}
        colorMode={colorMode}
      />
    </div>
  );
};