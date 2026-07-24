export const createReceiptRenderer = () => {
  const start = (context = {}) => {
    const kernel = context?.kernel || context?.runtime?.kernel;
    const mount = document.querySelector('.observatory-receipt');
    if (!mount || !kernel) return { stop() {} };
    const render = receipt => {
      mount.dataset.claimClass = receipt.claimClass;
      mount.dataset.receiptId = receipt.id;
      const label = mount.querySelector('span');
      const title = mount.querySelector('strong');
      const detail = mount.querySelector('small');
      if (label) label.textContent = `RECEIPT / ${receipt.id}`;
      if (title) title.textContent = `${receipt.claimClass.replaceAll('-', ' ')} · ${receipt.state}`;
      if (detail) detail.textContent = `${receipt.decision} · ${receipt.evidenceScope}`;
    };
    const stop = () => kernel.off?.('receipt', render);
    kernel.on('receipt', render);
    const current = kernel.state.receipts.at(-1);
    if (current) render(current);
    return { stop };
  };

  return { start };
};
