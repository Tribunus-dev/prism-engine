
export const createPersistentObjectSystem = () => {
  const start = (context) => {
    const kernel = context?.kernel;
    const kernelSubjectId = kernel?.subject?.id || kernel?.state?.subjectId || 'computational-subject:prism-model';
    kernel?.record?.({
      type: 'persistent-object-adapted-to-canonical-object',
      visible: 'compatibility naming layer preserved',
      transformed: 'persistent-object delegates to canonical object system',
      hidden: 'legacy persistence panel removed from direct rendering',
      subject: kernelSubjectId,
    });

    return { stop() {} };
  };

  return { start };
};
