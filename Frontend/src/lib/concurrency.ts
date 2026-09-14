/**
 * Bounded-concurrency map — the client-side half of SRS PE-2 ("Submitting a
 * batch of many files shall not block the interface").
 *
 * Before this existed, whole datasets were materialised with
 * `Promise.all(files.map(materializeAudio))`: 144 simultaneous POSTs for
 * RAVDESS, each of which makes the API probe, hash and copy a file. Over HTTP/2
 * the browser's six-connections-per-host limit does not apply, so all of them
 * reached the server at once, competing with every other user's requests; and
 * when one failed, `Promise.all` rejected but the remaining requests kept
 * running for nothing.
 *
 * `mapWithConcurrency` keeps at most `limit` calls in flight, returns results in
 * input order (callers join them positionally — see use-layer-probes.ts), and
 * starts no new call after the first failure.
 */
export async function mapWithConcurrency<T, R>(
  items: readonly T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  if (!Number.isInteger(limit) || limit < 1) {
    throw new RangeError(`concurrency limit must be a positive integer, got ${limit}`);
  }
  const results = new Array<R>(items.length);
  let next = 0;
  let failed = false;

  const worker = async () => {
    while (!failed && next < items.length) {
      const index = next++;
      try {
        results[index] = await fn(items[index], index);
      } catch (error) {
        failed = true;
        throw error;
      }
    }
  };

  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return results;
}
