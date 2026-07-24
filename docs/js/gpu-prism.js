import { runtimeContext } from './runtime/runtime-context.js';
import { createPrismError, ERROR_CODES } from './runtime/errors.js';

export const createGpuPrismSystem = () => {
  const start = (context = runtimeContext()) => {
    const kernel = context.kernel;
    const domRuntime = context.domRuntime;
    const config = context.config || {};
    if (config.gpu === false) return { stop() {} };
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return { stop() {} };

    const params = new URLSearchParams(location.search);
    if (params.get('prismGpu') === 'off') return { stop() {} };
    if (config.runtime === false) {
      throw createPrismError(ERROR_CODES.RENDERER_MOUNT_FAILED, 'Runtime was disabled for GPU system');
    }

    const createOrRecover = () => {
      const shellHost = document.querySelector('#prism-portal-root')
        || document.querySelector('.observatory-shell')
        || document.querySelector('main')
        || document.body;
      const shell = (() => {
        const existing = document.querySelector('#prism-effects-shell');
        if (existing) {
          domRuntime?.claim?.('gpu-prism-shell', '#prism-effects-shell');
          return existing;
        }
        const created = document.createElement('div');
        created.id = 'prism-effects-shell';
        created.setAttribute('aria-hidden', 'true');
        domRuntime?.claimNode?.('gpu-prism-shell', created);
        shellHost.append(created);
        return created;
      })();
      const root = (() => {
        const existing = document.querySelector('#prism-effects-root');
        if (existing) {
          if (existing.parentElement !== shell) {
            shell.append(existing);
          }
          domRuntime?.claimNode?.('gpu-prism-root', existing);
          return existing;
        }
        const created = document.createElement('div');
        created.id = 'prism-effects-root';
        created.setAttribute('aria-hidden', 'true');
        domRuntime?.claimNode?.('gpu-prism-root', created);
        shell.append(created);
        return created;
      })();

      return { shell, root };
    };

    const { shell, root } = createOrRecover();

    if (!shell.isConnected) {
      const shellHost = document.querySelector('#prism-portal-root')
        || document.querySelector('.observatory-shell')
        || document.querySelector('main')
        || document.body;
      shellHost.append(shell);
    }
    if (!root.isConnected || root.parentElement !== shell) {
      if (root.parentElement && root.parentElement !== shell) {
        root.remove();
      }
      shell.append(root);
    }

    domRuntime?.claim?.('gpu-prism-shell', '#prism-effects-shell');
    domRuntime?.claim?.('gpu-prism-root', '#prism-effects-root');

    const canvas = document.createElement('canvas');
    canvas.className = 'gpu-prism-field';
    canvas.setAttribute('aria-hidden', 'true');
    canvas.id = 'prism-effects-canvas';
    root.append(canvas);
    domRuntime?.claimNode?.('gpu-prism-canvas', canvas);
    domRuntime?.assertOwnership?.('gpu-prism-shell', shell);
    domRuntime?.assertOwnership?.('gpu-prism-root', root);
    domRuntime?.mark?.('gpu-prism-mounted', {
      shellConnected: shell.isConnected,
      rootConnected: root.isConnected,
      canvasSize: { width: canvas.width, height: canvas.height },
    });

    const gl = canvas.getContext('webgl', {
      alpha: true,
      antialias: false,
      powerPreference: 'high-performance',
    });
    if (!gl) {
      canvas.remove();
      if (domRuntime?.detectExternalProjectionEnvironment) {
        const external = domRuntime.detectExternalProjectionEnvironment();
        domRuntime.mark?.('gpu-context-unavailable', {
          owner: 'gpu-prism',
          external: external.present,
          activeNodes: external.activeNodes,
        });
      }
      throw createPrismError(ERROR_CODES.RENDERER_MOUNT_FAILED, 'WebGL unavailable for prism prism-gpu rendering');
    }

    const vertexSource = `attribute vec2 position; void main(){ gl_Position=vec4(position,0.0,1.0); }`;
    const fragmentSource = `
      precision highp float;
      uniform vec2 resolution;
      uniform vec2 pointer;
      uniform float time;
      uniform float scroll;
      uniform float stage;
      uniform float layer;

      float hash(vec2 p){ return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
      vec3 spectral(float angle, float emphasis){
        vec3 blue=vec3(.18,.35,1.0), violet=vec3(.66,.24,1.0), green=vec3(.12,1.0,.48), amber=vec3(1.0,.56,.13);
        float phase=fract(angle/6.28318+.5+stage*.07+time*.000008);
        vec3 color=mix(blue,violet,smoothstep(.05,.34,phase));
        color=mix(color,green,smoothstep(.34,.62,phase));
        color=mix(color,amber,smoothstep(.62,.88,phase));
        return color*emphasis;
      }
      void main(){
        vec2 uv=(gl_FragCoord.xy-.5*resolution)/min(resolution.x,resolution.y);
        uv-=pointer*.10;
        float distanceFromCenter=length(uv);
        float angle=atan(uv.y,uv.x);
        float motion=time*.000045+scroll*.000002;
        float stagePulse=0.72+0.28*sin(motion*2.0+stage*.9);
        float beam=exp(-24.0*pow(abs(uv.x+uv.y*.28+sin(motion)*.08),2.0))*exp(-1.35*distanceFromCenter);
        float refraction=pow(max(0.0,sin(angle*3.0+distanceFromCenter*12.0-motion*2.0+layer*.8)),16.0)*exp(-2.4*distanceFromCenter);
        float caustic=pow(max(0.0,sin(angle*5.0-distanceFromCenter*24.0+motion*1.7)),12.0)*exp(-8.0*abs(distanceFromCenter-.34));
        vec2 cell=floor((uv+1.0)*34.0);
        vec2 local=fract((uv+1.0)*34.0)-.5;
        float particles=step(.994,hash(cell))*smoothstep(.45,.02,length(local))*exp(-1.6*distanceFromCenter)*smoothstep(.16,.72,beam+refraction);
        vec3 color=spectral(angle,refraction*.65*stagePulse)+vec3(.7,.86,1.0)*beam*.22+spectral(angle+layer,caustic*.22);
        float alpha=(beam*.14+refraction*.055+caustic*.10+particles*.32)*smoothstep(1.2,.16,distanceFromCenter);
        gl_FragColor=vec4(color,alpha);
      }
    `;

    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      return gl.getShaderParameter(shader, gl.COMPILE_STATUS) ? shader : null;
    };

    const vertex = compile(gl.VERTEX_SHADER, vertexSource);
    const fragment = compile(gl.FRAGMENT_SHADER, fragmentSource);
    if (!vertex || !fragment) {
      canvas.remove();
      throw createPrismError(ERROR_CODES.RENDERER_MOUNT_FAILED, 'GPU shader compilation failed');
    }

    const program = gl.createProgram();
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      canvas.remove();
      throw createPrismError(ERROR_CODES.RENDERER_MOUNT_FAILED, 'GPU shader link failed');
    }
    gl.useProgram(program);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW);
    const position = gl.getAttribLocation(program, 'position');
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

    const uniforms = Object.fromEntries(['resolution','pointer','time','scroll','stage','layer'].map(name => [name, gl.getUniformLocation(program, name)]));
    let pointer = { x: 0, y: 0 };
    const layerValue = () => ({ logical: 0.2, physical: 1.4, execution: 2.8 }[document.body.dataset.prismLayer] || 0);
    const stageValue = () => Number(document.body.dataset.prismStage || 1);

    const pointerMove = event => {
      pointer = { x: event.clientX / innerWidth - .5, y: .5 - event.clientY / innerHeight };
    };
    const resize = () => {
      const density = Math.min(devicePixelRatio || 1, 2);
      canvas.width = innerWidth * density;
      canvas.height = innerHeight * density;
      canvas.style.width = `${innerWidth}px`;
      canvas.style.height = `${innerHeight}px`;
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform2f(uniforms.resolution, canvas.width, canvas.height);
    };
    const update = event => {
      gl.uniform1f(uniforms.time, event.timeStamp || performance.now());
      gl.uniform1f(uniforms.scroll, scrollY);
      gl.uniform2f(uniforms.pointer, pointer.x, pointer.y);
      gl.uniform1f(uniforms.stage, stageValue());
      gl.uniform1f(uniforms.layer, layerValue());
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    };

    let frame;
    const draw = now => {
      update({ timeStamp: now });
      frame = requestAnimationFrame(draw);
    };

    const scrollHandler = () => {
      if (kernel) {
        kernel.emit('optical-effect', {
          type: 'prism-gpu-pulse',
          stage: document.body.dataset.prismStage || 'source',
          layer: document.body.dataset.prismLayer || 'logical',
          progress: document.body.style.getPropertyValue('--prism-scroll-progress') || '0',
        });
      }
    };

    window.addEventListener('pointermove', pointerMove, { passive: true });
    window.addEventListener('resize', resize);
    window.addEventListener('scroll', scrollHandler, { passive: true });
    kernel?.on?.('scroll', scrollHandler);
    kernel?.on?.('optical-state', scrollHandler);

    resize();
    frame = requestAnimationFrame(draw);
    update({ timeStamp: performance.now() });

    const stop = () => {
      cancelAnimationFrame(frame);
      domRuntime?.mark?.('gpu-prism-stop', {
        canvasSize: { width: canvas.width, height: canvas.height },
      });
      window.removeEventListener('pointermove', pointerMove);
      window.removeEventListener('resize', resize);
      window.removeEventListener('scroll', scrollHandler);
      kernel?.off?.('scroll', scrollHandler);
      kernel?.off?.('optical-state', scrollHandler);
      if (canvas.isConnected) {
        canvas.remove();
      }
      const shell = document.querySelector('#prism-effects-shell');
      const mountRoot = document.querySelector('#prism-effects-root');
      const shellOwner = shell?.getAttribute?.('data-prism-owned');
      const rootOwner = mountRoot?.getAttribute?.('data-prism-owned');
      if (mountRoot && mountRoot.children.length === 0 && (!rootOwner || rootOwner === 'gpu-prism-root')) {
        mountRoot.remove();
      }
      if (shell && shell.children.length === 0 && (!shellOwner || shellOwner === 'gpu-prism-shell')) {
        shell.remove();
      }
    };

    return { stop };
  };

  return { start };
};
