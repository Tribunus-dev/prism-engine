export const parseRuntimeFlag = (value, fallback) => {
  if (value === null || value === undefined || value === '') return fallback;
  if (['0', 'false', 'off', 'no', 'disabled'].includes(String(value).toLowerCase())) return false;
  if (['1', 'true', 'on', 'yes', 'enabled'].includes(String(value).toLowerCase())) return true;
  return fallback;
};

export const createRuntimeConfig = (search = '') => {
  const params = new URLSearchParams(search);
  const readBool = (name, fallback = false, alias) => {
    const value = params.get(name);
    if (value !== null && value !== undefined) return parseRuntimeFlag(value, fallback);
    if (alias) {
      const aliasValue = params.get(alias);
      if (aliasValue !== null && aliasValue !== undefined) return parseRuntimeFlag(aliasValue, fallback);
    }
    return fallback;
  };

  return {
    debug: parseRuntimeFlag(params.get('prismDebug'), false),
    diagnostics: parseRuntimeFlag(params.get('prismDiagnostics'), false),
    observer: readBool('prismObserver', true),
    gpu: readBool('prismGpu', true),
    shell: readBool('prismShell', true),
    receipts: readBool('prismReceipts', true),
    navigation: readBool('prismNavigation', true),
    scroll: readBool('prismScroll', true),
    accessibility: readBool('prismAccessibility', true),
    continuity: readBool('prismContinuity', true),
    effects: readBool('prismEffects', true),
    runtime: parseRuntimeFlag(params.get('prismRuntime'), true),
    runtimeHardAbort: parseRuntimeFlag(params.get('prismRuntimeHardAbort'), false),
    systems: params.get('prismSystems') || '',
    runtimeDiagnostics: parseRuntimeFlag(params.get('prismDiagnostics'), false),
    runtimeObserver: parseRuntimeFlag(params.get('prismObserver'), true),
    prismDiagnostics: parseRuntimeFlag(params.get('prismDiagnostics'), false),
    prismObserver: parseRuntimeFlag(params.get('prismObserver'), true),
    prismGpu: parseRuntimeFlag(params.get('prismGpu'), true),
    prismShell: parseRuntimeFlag(params.get('prismShell'), true),
    prismReceipts: parseRuntimeFlag(params.get('prismReceipts'), true),
    prismNavigation: parseRuntimeFlag(params.get('prismNavigation'), true),
    prismScroll: parseRuntimeFlag(params.get('prismScroll'), true),
    prismAccessibility: parseRuntimeFlag(params.get('prismAccessibility'), true),
    prismContinuity: parseRuntimeFlag(params.get('prismContinuity'), true),
    prismEffects: parseRuntimeFlag(params.get('prismEffects'), true),
  };
};
