import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { HelpCircle } from "lucide-react";

interface ChartCardProps {
  title: string;
  explanation?: string;
  children: React.ReactNode;
  // "y" = drag-resize taller only (default); "both" = also widen, for the
  // correlation matrix which needs more room to fit legible labels.
  resize?: "y" | "both";
  // Extra controls rendered on the right of the title row (e.g. a legend toggle).
  actions?: React.ReactNode;
}

export const ChartCard = ({ title, explanation, children, resize = "y", actions }: ChartCardProps) => (
  <div className="border border-border rounded-lg bg-card p-2">
    <div className="flex items-center gap-1.5 mb-1 px-1">
      <span className="text-xs font-medium text-foreground">{title}</span>
      {explanation && (
        <Tooltip>
          <TooltipTrigger>
            <HelpCircle className="h-3 w-3 text-muted-foreground" />
          </TooltipTrigger>
          <TooltipContent className="max-w-xs text-xs">{explanation}</TooltipContent>
        </Tooltip>
      )}
      {actions && <div className="ml-auto flex items-center gap-1">{actions}</div>}
    </div>
    <div
      className={`h-56 min-h-[10rem] max-h-[36rem] overflow-hidden ${resize === "both" ? "resize max-w-full" : "resize-y"}`}
    >
      {children}
    </div>
  </div>
);
