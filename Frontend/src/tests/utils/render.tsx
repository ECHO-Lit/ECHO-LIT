/**
 * Provider harness for §3.1.3 component cases.
 *
 * Reproduces App.tsx's provider stack MINUS BrowserRouter. BrowserRouter lives
 * inside App.tsx, not main.tsx, so route-level cases render the real <App/> and
 * drive jsdom's real history instead (see app-navigation.test.tsx); component
 * cases that only need <Link> or useLocation opt into a MemoryRouter here.
 *
 * A FRESH QueryClient per call is mandatory, not stylistic. use-job-query.ts
 * sets `gcTime: Infinity` on two key families precisely so long jobs survive.
 * A client shared across tests would therefore carry one test's job status into
 * the next and produce order-dependent passes — the exact failure mode 3.1.2's
 * Special Consideration 5 describes for lru_cache.
 */
import { type ReactElement, type ReactNode } from "react";
import { render, type RenderOptions, type RenderResult } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { EmbeddingProvider } from "@/contexts/EmbeddingContext";

export interface ProviderOptions extends Omit<RenderOptions, "wrapper"> {
  /** Mount a MemoryRouter at this entry. Omit for components with no routing. */
  route?: string;
  /** Six components throw without it; one case asserts that throw. */
  withEmbeddingProvider?: boolean;
  /** Mount the sonner viewport so toast assertions are real. Default true. */
  withToaster?: boolean;
}

export function createTestQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0, staleTime: 0 },
      mutations: { retry: false },
    },
  });
}

export function renderWithProviders(
  ui: ReactElement,
  options: ProviderOptions = {},
): RenderResult & { queryClient: QueryClient } {
  const {
    route,
    withEmbeddingProvider = false,
    withToaster = true,
    ...renderOptions
  } = options;
  const queryClient = createTestQueryClient();

  const Wrapper = ({ children }: { children: ReactNode }) => {
    let tree = <>{children}</>;
    if (withEmbeddingProvider) tree = <EmbeddingProvider>{tree}</EmbeddingProvider>;
    if (route !== undefined) {
      tree = <MemoryRouter initialEntries={[route]}>{tree}</MemoryRouter>;
    }
    return (
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>
          {withToaster && <Sonner />}
          {tree}
        </TooltipProvider>
      </QueryClientProvider>
    );
  };

  return {
    ...render(ui, { wrapper: Wrapper, ...renderOptions }),
    queryClient,
  };
}
