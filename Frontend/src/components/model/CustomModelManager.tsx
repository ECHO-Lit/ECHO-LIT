import { useCallback, useEffect, useState } from 'react';
import { AlertCircle, BrainCircuit, CheckCircle2, LoaderCircle, Plus, RefreshCw, Trash2 } from 'lucide-react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  CustomModel,
  deleteCustomModel,
  listCustomModels,
  registerCustomModel,
} from '@/lib/models';

interface CustomModelManagerProps {
  onModelsChanged?: (models: CustomModel[]) => void;
}

const kindLabel: Record<NonNullable<CustomModel['kind']>, string> = {
  seq2seq_asr: 'Seq2Seq ASR',
  ctc_asr: 'CTC ASR',
  audio_classification: 'Audio classification',
};

export const CustomModelManager = ({ onModelsChanged }: CustomModelManagerProps) => {
  const [open, setOpen] = useState(false);
  const [models, setModels] = useState<CustomModel[]>([]);
  const [hfRepo, setHfRepo] = useState('');
  const [revision, setRevision] = useState('');
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);

  const loadModels = useCallback(async () => {
    setLoading(true);
    try {
      const next = await listCustomModels();
      setModels(next);
      onModelsChanged?.(next);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Unable to load custom models');
    } finally {
      setLoading(false);
    }
  }, [onModelsChanged]);

  useEffect(() => {
    if (open) void loadModels();
  }, [open, loadModels]);

  useEffect(() => {
    if (!open || !models.some((model) => model.status === 'validating')) return;
    const timer = window.setInterval(() => void loadModels(), 2000);
    return () => window.clearInterval(timer);
  }, [open, models, loadModels]);

  const handleRegister = async () => {
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(hfRepo.trim())) {
      toast.error('Use a Hugging Face repository in owner/model form.');
      return;
    }
    setSubmitting(true);
    try {
      await registerCustomModel(hfRepo.trim(), revision);
      setHfRepo('');
      setRevision('');
      toast.success('Model submitted for validation');
      await loadModels();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Unable to register model');
    } finally {
      setSubmitting(false);
    }
  };

  const handleDelete = async (model: CustomModel) => {
    setDeleting(model.model_id);
    try {
      await deleteCustomModel(model.model_id);
      toast.success(`${model.hf_repo} removed`);
      await loadModels();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Unable to remove model');
    } finally {
      setDeleting(null);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="h-7 text-xs">
          <BrainCircuit className="h-3.5 w-3.5 mr-1.5" />
          Models
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Custom Hugging Face models</DialogTitle>
          <DialogDescription>
            Register a standard audio model by repository ID. Validation runs in a worker with remote model code disabled.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-3 rounded-md border border-border p-3 sm:grid-cols-[1fr_10rem_auto]">
          <div className="space-y-1">
            <Label htmlFor="hf-repository" className="text-xs">Repository</Label>
            <Input
              id="hf-repository"
              value={hfRepo}
              onChange={(event) => setHfRepo(event.target.value)}
              placeholder="openai/whisper-tiny"
              className="h-8 text-xs"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="hf-revision" className="text-xs">Revision (optional)</Label>
            <Input
              id="hf-revision"
              value={revision}
              onChange={(event) => setRevision(event.target.value)}
              placeholder="main"
              className="h-8 text-xs"
            />
          </div>
          <Button onClick={() => void handleRegister()} disabled={submitting} className="self-end h-8 text-xs">
            {submitting ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5 mr-1" />}
            Add model
          </Button>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-xs font-medium">Registered models</span>
          <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => void loadModels()} disabled={loading}>
            <RefreshCw className={`h-3.5 w-3.5 mr-1 ${loading ? 'animate-spin' : ''}`} /> Refresh
          </Button>
        </div>

        {models.length === 0 && !loading ? (
          <div className="rounded-md border border-dashed border-border p-6 text-center text-xs text-muted-foreground">
            No custom models registered in this session.
          </div>
        ) : (
          <div className="space-y-2">
            {models.map((model) => (
              <div key={model.model_id} className="rounded-md border border-border p-3 space-y-2">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="font-mono text-xs truncate">{model.hf_repo}</div>
                    {model.revision && <div className="text-[10px] text-muted-foreground">revision: {model.revision}</div>}
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    {model.status === 'ready' && <Badge className="bg-emerald-600 hover:bg-emerald-600"><CheckCircle2 className="h-3 w-3 mr-1" />Ready</Badge>}
                    {model.status === 'validating' && <Badge variant="secondary"><LoaderCircle className="h-3 w-3 mr-1 animate-spin" />Validating</Badge>}
                    {model.status === 'failed' && <Badge variant="destructive"><AlertCircle className="h-3 w-3 mr-1" />Failed</Badge>}
                    <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => void handleDelete(model)} disabled={deleting === model.model_id} aria-label={`Remove ${model.hf_repo}`}>
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>
                {model.status === 'ready' && (
                  <div className="flex flex-wrap gap-1">
                    {model.kind && <Badge variant="outline" className="text-[10px]">{kindLabel[model.kind]}</Badge>}
                    {model.capabilities.map((capability) => <Badge key={capability} variant="secondary" className="text-[10px]">{capability.replace('_', ' ')}</Badge>)}
                  </div>
                )}
                {model.status === 'failed' && <div className="text-xs text-destructive">{model.error || 'Validation failed.'}</div>}
              </div>
            ))}
          </div>
        )}

        <p className="text-[11px] text-muted-foreground">
          Registered models expire with the session. Job execution and model-selection support will become available once the generic worker adapter is connected.
        </p>
      </DialogContent>
    </Dialog>
  );
};
