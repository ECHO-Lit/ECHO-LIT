import React from "react";
import { Button } from "@/components/ui/button";
import { AlertTriangle } from "lucide-react";

interface ErrorBoundaryProps {
  children: React.ReactNode;
  /** Names the region that failed, so a scoped boundary can say what is lost. */
  label?: string;
}

interface ErrorBoundaryState {
  error: Error | null;
}

/**
 * The application's only React error boundary.
 *
 * Without one, a single render-time throw — a malformed job result, an
 * unexpected null, or a context consumer mounted outside its provider —
 * unmounts the entire tree and leaves a blank white page with no message and no
 * way back. SRS US-4 requires errors to be "clear, actionable messages in plain
 * language rather than raw stack traces", which is exactly what an unhandled
 * throw fails to provide.
 *
 * The raw error text is deliberately NOT rendered: it is a developer artefact,
 * not plain language. It is logged to the console for diagnosis instead.
 */
export class ErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("Unhandled render error", error, info.componentStack);
  }

  private handleReload = () => {
    window.location.reload();
  };

  render() {
    if (!this.state.error) return this.props.children;

    const scope = this.props.label
      ? `The ${this.props.label} could not be displayed.`
      : "This view could not be displayed.";

    return (
      <div
        role="alert"
        className="flex h-full w-full flex-col items-center justify-center gap-3 p-6 text-center"
      >
        <AlertTriangle className="h-6 w-6 text-destructive" aria-hidden="true" />
        <h2 className="text-sm font-semibold">Something went wrong</h2>
        <p className="max-w-sm text-xs text-muted-foreground">
          {scope} Your uploaded audio and running jobs are unaffected. Reloading
          the page usually clears this; if it happens again, try a different file
          or analysis.
        </p>
        <Button size="sm" variant="outline" onClick={this.handleReload}>
          Reload the page
        </Button>
      </div>
    );
  }
}

export default ErrorBoundary;
