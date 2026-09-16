import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import JacobianLensLab from "./pages/JacobianLensLab";

const queryClient = new QueryClient();

// Only one toast system is mounted. The Radix <Toaster/> that used to sit here
// was invoked by no feature code anywhere in src/ — every notification in the
// product goes through sonner — while still mounting a permanent live region
// and shipping TOAST_LIMIT = 1 with TOAST_REMOVE_DELAY = 1000000 ms.
const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Sonner />
      <BrowserRouter>
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/j-lens" element={<JacobianLensLab />} />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </ErrorBoundary>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
