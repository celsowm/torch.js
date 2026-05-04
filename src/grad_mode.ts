/**
 * Autograd context managers.
 * @pytorch torch.no_grad(), torch.enable_grad(), torch.inference_mode()
 */

let _grad_mode = true;
let _inference_mode = false;

/**
 * Returns whether gradient tracking is currently enabled.
 */
export const is_grad_enabled = (): boolean => _grad_mode && !_inference_mode;

/**
 * Context manager that disables gradient tracking.
 * @pytorch torch.no_grad()
 *
 * @example
 * ```ts
 * with no_grad():
 *   y = model(x)
 * ```
 */
export class no_grad {
  private prev: boolean;

  constructor() {
    this.prev = _grad_mode;
  }

  [Symbol.dispose](): void {
    _grad_mode = this.prev;
  }

  /**
   * Use as decorator: no_grad.decorate(fn)
   */
  static decorate<T extends (...args: any[]) => any>(fn: T): T {
    return ((...args: any[]) => {
      const ctx = new no_grad();
      try {
        return fn(...args);
      } finally {
        ctx[Symbol.dispose]();
      }
    }) as T;
  }

  /**
   * Use as a wrapper: no_grad.run(() => ...)
   */
  static run<T>(fn: () => T): T {
    const ctx = new no_grad();
    try {
      return fn();
    } finally {
      ctx[Symbol.dispose]();
    }
  }
}

/**
 * Context manager that enables gradient tracking.
 * @pytorch torch.enable_grad()
 */
export class enable_grad {
  private prev: boolean;

  constructor() {
    this.prev = _grad_mode;
    _grad_mode = true;
  }

  [Symbol.dispose](): void {
    _grad_mode = this.prev;
  }

  static decorate<T extends (...args: any[]) => any>(fn: T): T {
    return ((...args: any[]) => {
      const ctx = new enable_grad();
      try {
        return fn(...args);
      } finally {
        ctx[Symbol.dispose]();
      }
    }) as T;
  }

  static run<T>(fn: () => T): T {
    const ctx = new enable_grad();
    try {
      return fn();
    } finally {
      ctx[Symbol.dispose]();
    }
  }
}

/**
 * Context manager that enables inference mode (no gradient tracking, no gradient history).
 * @pytorch torch.inference_mode()
 */
export class inference_mode {
  private prevMode: boolean;
  private prevInference: boolean;

  constructor() {
    this.prevMode = _grad_mode;
    this.prevInference = _inference_mode;
    _grad_mode = false;
    _inference_mode = true;
  }

  [Symbol.dispose](): void {
    _grad_mode = this.prevMode;
    _inference_mode = this.prevInference;
  }

  static decorate<T extends (...args: any[]) => any>(fn: T): T {
    return ((...args: any[]) => {
      const ctx = new inference_mode();
      try {
        return fn(...args);
      } finally {
        ctx[Symbol.dispose]();
      }
    }) as T;
  }

  static run<T>(fn: () => T): T {
    const ctx = new inference_mode();
    try {
      return fn();
    } finally {
      ctx[Symbol.dispose]();
    }
  }
}

/**
 * Decorator that disables gradient tracking during a function call.
 *
 * @example
 * ```ts
 * class MyModel extends Module {
 *   \@no_grad_decorator
 *   predict(x: Tensor): Tensor { ... }
 * }
 * ```
 */
export function no_grad_decorator(
  target: any,
  propertyKey: string,
  descriptor: PropertyDescriptor
): PropertyDescriptor {
  const originalMethod = descriptor.value;
  descriptor.value = function (...args: any[]) {
    return no_grad.run(() => originalMethod.apply(this, args));
  };
  return descriptor;
}
