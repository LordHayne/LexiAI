class VoiceCaptureProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const config = options && options.processorOptions ? options.processorOptions : {};
        this.bufferSize = config.bufferSize || 4096;
        this.buffer = new Float32Array(this.bufferSize);
        this.offset = 0;
    }

    process(inputs) {
        const input = inputs[0];
        if (input && input[0]) {
            const channel = input[0];
            let i = 0;
            while (i < channel.length) {
                const remaining = this.bufferSize - this.offset;
                const copyCount = Math.min(remaining, channel.length - i);
                this.buffer.set(channel.subarray(i, i + copyCount), this.offset);
                this.offset += copyCount;
                i += copyCount;

                if (this.offset === this.bufferSize) {
                    this.port.postMessage(this.buffer, [this.buffer.buffer]);
                    this.buffer = new Float32Array(this.bufferSize);
                    this.offset = 0;
                }
            }
        }
        return true;
    }
}

registerProcessor('voice-capture-processor', VoiceCaptureProcessor);
