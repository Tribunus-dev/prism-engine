const memoryKey = 'prism-observatory-continuity';

export const createContinuityService = ({ storage = localStorage } = {}) => {
  const read = () => { try { return JSON.parse(storage.getItem(memoryKey) || '{}'); } catch { return {}; } };
  return {
    initial() { return read(); },
    save(update) { try { storage.setItem(memoryKey, JSON.stringify(update)); } catch {} return update; },
    reset() { try { storage.removeItem(memoryKey); } catch {} return { visits: 0, lastObservation: null, lastStage: null }; }
  };
};
