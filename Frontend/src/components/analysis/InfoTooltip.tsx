import { Info } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

interface InfoTooltipProps {
  text: string;
  className?: string;
}

/** Small (i) icon that reveals a glossary definition on hover/focus.
 * Relies on the TooltipProvider already mounted at the app root (App.tsx). */
export function InfoTooltip({ text, className }: InfoTooltipProps) {
  return (
    <Tooltip delayDuration={150}>
      <TooltipTrigger asChild>
        <button
          type="button"
          className={`inline-flex items-center text-muted-foreground hover:text-foreground focus:outline-none focus-visible:ring-1 focus-visible:ring-ring rounded-sm ${className ?? ""}`}
          aria-label="More information"
        >
          <Info className="h-3.5 w-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs text-xs leading-relaxed">
        {text}
      </TooltipContent>
    </Tooltip>
  );
}
