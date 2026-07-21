import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  AlertCircle,
  Brain,
  CheckCircle,
  HelpCircle,
  MinusCircle,
  Plus,
  Trash2,
  X,
} from 'lucide-react';
import {
  ConstraintCheck,
  CustomModelSpec,
  ValidationResult,
  deleteCustomModel,
  fetchCustomModelCapabilities,
  fetchCustomModels,
  registerCustomModel,
  validateCustomModel,
} from '@/lib/customModels';

interface CustomModelManagerProps {
  onModelRegistered?: (formattedName: string) => void;
  onModelSelected?: (formattedName: string) => void;
  onModelsChanged?: () => void;
}

const statusIcon = (status: ConstraintCheck['status']) => {
  if (status === 'pass') return <CheckCircle className="h-4 w-4 text-green-600 shrink-0" />;
  if (status === 'fail') return <AlertCircle className="h-4 w-4 text-red-600 shrink-0" />;
  if (status === 'warn') return <AlertCircle className="h-4 w-4 text-amber-500 shrink-0" />;
  return <MinusCircle className="h-4 w-4 text-muted-foreground shrink-0" />;
};

const ConstraintReport = ({ checks }: { checks: ConstraintCheck[] }) => (
  <div className="space-y-1.5">
    {checks.map((check) => (
      <div key={check.id} className="flex items-start gap-2 text-xs">
        {statusIcon(check.status)}
        <div className="min-w-0">
          <div className="font-medium">{check.constraint}</div>
          <div className="text-muted-foreground break-words">{check.detail}</div>
        </div>
      </div>
    ))}
  </div>
);

export const CustomModelManager: React.FC<CustomModelManagerProps> = ({
  onModelRegistered,
  onModelSelected,
  onModelsChanged,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('list');
  const [models, setModels] = useState<CustomModelSpec[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Add-model form
  const [modelId, setModelId] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [revision, setRevision] = useState('');
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [checking, setChecking] = useState(false);
  const [registering, setRegistering] = useState(false);
  const [failedReport, setFailedReport] = useState<ConstraintCheck[] | null>(null);

  // Support matrix, rendered from the backend so the contract lives in one place
  const [support, setSupport] = useState<any>(null);

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      setModels(await fetchCustomModels());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load custom models');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!isOpen) return;
    refresh();
    fetchCustomModelCapabilities().then(setSupport).catch(() => setSupport(null));
  }, [isOpen]);

  const runCheck = async () => {
    if (!modelId.trim()) {
      setError('A Hugging Face model ID is required');
      return;
    }
    setChecking(true);
    setError(null);
    setValidation(null);
    setFailedReport(null);
    try {
      // Shallow check: config + processor only, so the user gets an answer in
      // about a second instead of waiting for a multi-GB weight download.
      setValidation(await validateCustomModel(modelId.trim(), false, revision.trim()));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Compatibility check failed');
    } finally {
      setChecking(false);
    }
  };

  const register = async () => {
    if (!modelId.trim()) {
      setError('A Hugging Face model ID is required');
      return;
    }
    setRegistering(true);
    setError(null);
    setFailedReport(null);
    try {
      const name = displayName.trim() || modelId.trim().split('/').pop() || modelId.trim();
      const result = await registerCustomModel(name, modelId.trim(), revision.trim());
      setModelId('');
      setDisplayName('');
      setRevision('');
      setValidation(null);
      await refresh();
      onModelsChanged?.();
      onModelRegistered?.(result.model);
      setActiveTab('list');
    } catch (err) {
      const e = err as Error & { report?: { checks: ConstraintCheck[] } };
      setError(e.message);
      if (e.report?.checks) setFailedReport(e.report.checks);
    } finally {
      setRegistering(false);
    }
  };

  const remove = async (name: string) => {
    if (!confirm(`Remove the custom model "${name}" from this session?`)) return;
    try {
      await deleteCustomModel(name);
      await refresh();
      onModelsChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete model');
    }
  };

  const select = (formattedName: string) => {
    onModelSelected?.(formattedName);
    setIsOpen(false);
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="h-7 text-xs">
          <Brain className="h-4 w-4 mr-2" />
          Custom Models
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-4xl max-h-[85vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle>Custom Model Manager</DialogTitle>
          <DialogDescription>
            Add Hugging Face audio models by repository ID. ECHO supports models by task and
            architecture, not by name.
          </DialogDescription>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="list" className="flex items-center gap-2">
              <Brain className="h-4 w-4" />
              My Models
            </TabsTrigger>
            <TabsTrigger value="add" className="flex items-center gap-2">
              <Plus className="h-4 w-4" />
              Add Model
            </TabsTrigger>
            <TabsTrigger value="support" className="flex items-center gap-2">
              <HelpCircle className="h-4 w-4" />
              What's Supported
            </TabsTrigger>
          </TabsList>

          <div className="mt-4 max-h-[55vh] overflow-y-auto">
            {/* ── Registered models ─────────────────────────────── */}
            <TabsContent value="list" className="space-y-4">
              {loading && (
                <div className="text-center py-8">
                  <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
                  Loading models...
                </div>
              )}

              {!loading && models.length === 0 && (
                <div className="text-center py-8 text-muted-foreground">
                  <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No custom models registered</p>
                  <p className="text-sm">Add a Hugging Face model to get started</p>
                </div>
              )}

              {!loading &&
                models.map((m) => (
                  <Card key={m.name}>
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between gap-2">
                        <CardTitle className="text-base truncate">{m.name}</CardTitle>
                        <div className="flex items-center gap-2 shrink-0">
                          <Badge variant="secondary">{m.task_label}</Badge>
                          <Button variant="outline" size="sm" onClick={() => select(m.formatted_name)}>
                            Select
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => remove(m.name)}
                            className="text-red-600 hover:text-red-700"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                      <CardDescription className="truncate">
                        {m.model_id}
                        {m.revision ? ` @ ${m.revision}` : ''} · {m.architecture}
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="grid grid-cols-3 gap-3 text-xs">
                        <div>
                          <span className="font-medium">Parameters:</span>{' '}
                          {(m.num_parameters / 1e6).toFixed(0)}M
                        </div>
                        <div>
                          <span className="font-medium">Sampling rate:</span> {m.sampling_rate} Hz
                        </div>
                        <div>
                          <span className="font-medium">Input:</span> {m.input_name}
                        </div>
                      </div>

                      <div>
                        <p className="text-xs font-medium mb-1.5">Available analyses</p>
                        <div className="flex flex-wrap gap-1">
                          {m.capability_labels.map((label) => (
                            <Badge key={label} variant="outline" className="text-[10px]">
                              {label}
                            </Badge>
                          ))}
                        </div>
                      </div>

                      {m.id2label && (
                        <div className="text-xs">
                          <span className="font-medium">Labels ({m.num_labels}):</span>{' '}
                          <span className="text-muted-foreground">
                            {Object.values(m.id2label).slice(0, 8).join(', ')}
                            {(m.num_labels || 0) > 8 ? ' ...' : ''}
                          </span>
                        </div>
                      )}

                      {m.warnings.length > 0 && (
                        <div className="text-xs text-amber-600 space-y-0.5">
                          {m.warnings.map((w, i) => (
                            <div key={i} className="flex items-start gap-1.5">
                              <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-px" />
                              <span>{w}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
            </TabsContent>

            {/* ── Add a model ───────────────────────────────────── */}
            <TabsContent value="add" className="space-y-4">
              <div>
                <Label htmlFor="model-id">Hugging Face model ID</Label>
                <Input
                  id="model-id"
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  placeholder="e.g. facebook/wav2vec2-base-960h"
                  className="mt-1"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label htmlFor="display-name">Display name (optional)</Label>
                  <Input
                    id="display-name"
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    placeholder="defaults to the repo name"
                    className="mt-1"
                  />
                </div>
                <div>
                  <Label htmlFor="revision">Revision (optional)</Label>
                  <Input
                    id="revision"
                    value={revision}
                    onChange={(e) => setRevision(e.target.value)}
                    placeholder="branch, tag or commit"
                    className="mt-1"
                  />
                </div>
              </div>

              <div className="flex gap-2">
                <Button
                  variant="outline"
                  onClick={runCheck}
                  disabled={checking || registering || !modelId.trim()}
                  className="flex-1"
                >
                  {checking ? 'Checking...' : 'Check compatibility'}
                </Button>
                <Button
                  onClick={register}
                  disabled={registering || checking || !modelId.trim()}
                  className="flex-1"
                >
                  {registering ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                      Downloading and validating...
                    </>
                  ) : (
                    <>
                      <Plus className="h-4 w-4 mr-2" />
                      Add model
                    </>
                  )}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                "Check compatibility" reads only the repository config — fast, no weights. "Add
                model" downloads the weights and runs a smoke-test forward pass before registering.
              </p>

              {validation && (
                <Card>
                  <CardHeader className="pb-3">
                    <div className="flex items-center gap-2">
                      {validation.compatible ? (
                        <CheckCircle className="h-5 w-5 text-green-600" />
                      ) : (
                        <AlertCircle className="h-5 w-5 text-red-600" />
                      )}
                      <CardTitle className="text-base">
                        {validation.compatible
                          ? 'Config checks passed'
                          : 'Not compatible'}
                      </CardTitle>
                    </div>
                    {validation.task_label && (
                      <CardDescription>
                        {validation.task_label} · {validation.architecture} · loads with{' '}
                        {validation.auto_class}
                      </CardDescription>
                    )}
                    {validation.error && (
                      <CardDescription className="text-red-600">{validation.error}</CardDescription>
                    )}
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {validation.capability_labels && (
                      <div className="flex flex-wrap gap-1">
                        {validation.capability_labels.map((label) => (
                          <Badge key={label} variant="outline" className="text-[10px]">
                            {label}
                          </Badge>
                        ))}
                      </div>
                    )}
                    <ConstraintReport checks={validation.checks} />
                  </CardContent>
                </Card>
              )}

              {failedReport && (
                <Card className="border-red-200">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base text-red-700">Registration rejected</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ConstraintReport checks={failedReport} />
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            {/* ── Support matrix ────────────────────────────────── */}
            <TabsContent value="support" className="space-y-4">
              {!support && <p className="text-sm text-muted-foreground">Loading support matrix...</p>}
              {support && (
                <>
                  <div>
                    <h4 className="text-sm font-semibold mb-2">Requirements</h4>
                    <ul className="space-y-1">
                      {support.constraints.map((c: { id: string; text: string }) => (
                        <li key={c.id} className="flex items-start gap-2 text-xs">
                          <CheckCircle className="h-3.5 w-3.5 text-muted-foreground shrink-0 mt-0.5" />
                          <span>{c.text}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div>
                    <h4 className="text-sm font-semibold mb-2">Analyses by architecture</h4>
                    <div className="space-y-2">
                      {support.tasks.map((t: any) => (
                        <Card key={t.task}>
                          <CardHeader className="pb-2">
                            <CardTitle className="text-sm">{t.label}</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="flex flex-wrap gap-1">
                              {t.capabilities.map((c: { id: string; label: string }) => (
                                <Badge key={c.id} variant="outline" className="text-[10px]">
                                  {c.label}
                                </Badge>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-semibold mb-2">Limits</h4>
                    <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                      <div>Max parameters: {(support.limits.max_parameters / 1e6).toFixed(0)}M</div>
                      <div>
                        Max size: {(support.limits.max_storage_bytes / 1024 ** 3).toFixed(1)} GiB
                      </div>
                      <div>Max inference: {support.limits.max_inference_seconds}s per file</div>
                      <div>Max models per session: {support.limits.max_models_per_session}</div>
                    </div>
                  </div>
                </>
              )}
            </TabsContent>
          </div>
        </Tabs>

        {error && (
          <div className="mt-2 p-3 bg-red-50 border border-red-200 rounded-md flex items-start gap-2">
            <AlertCircle className="h-4 w-4 text-red-500 shrink-0 mt-0.5" />
            <span className="text-sm text-red-700">{error}</span>
            <Button variant="ghost" size="sm" onClick={() => setError(null)} className="ml-auto shrink-0">
              <X className="h-4 w-4" />
            </Button>
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => setIsOpen(false)}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
