def cleanup(self):
    self.ctx.destroy(linger=0)
    if model_executor := getattr(self.engine, "model_executor", None):
        model_executor.shutdown()