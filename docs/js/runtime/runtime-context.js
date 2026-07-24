let currentRuntime = null;

export const setRuntimeContext = runtime => { currentRuntime = runtime; };
export const runtimeContext = () => currentRuntime;
