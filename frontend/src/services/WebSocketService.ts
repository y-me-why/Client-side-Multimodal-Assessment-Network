class WebSocketService {
  private socket: WebSocket | null = null;
  private sessionId: string = '';
  private isConnected: boolean = false;
  private reconnectInterval: number = 5000; // 5 seconds
  private maxReconnectAttempts: number = 10;
  private reconnectAttempts: number = 0;

  // Event handlers
  private onFacialAnalysis: ((data: any) => void) | null = null;
  private onVoiceAnalysis: ((data: any) => void) | null = null;
  private onConnectionChange: ((isConnected: boolean) => void) | null = null;
  private onError: ((error: any) => void) | null = null;
  private onQuestionGenerated: ((data: any) => void) | null = null;
  private onResponseReceived: ((data: any) => void) | null = null;

  constructor() {
    this.connect = this.connect.bind(this);
    this.disconnect = this.disconnect.bind(this);
    this.sendFacialData = this.sendFacialData.bind(this);
    this.sendVoiceData = this.sendVoiceData.bind(this);
  }

  connect(sessionId: string): Promise<boolean> {
    this.sessionId = sessionId;
    const wsUrl = `ws://localhost:8001/ws/live-analysis/${sessionId}`;

    return new Promise((resolve) => {
      try {
        this.socket = new WebSocket(wsUrl);

        this.socket.onopen = () => {
          console.log('WebSocket connected for live analysis');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this.onConnectionChange?.(true);
          resolve(true);
        };

        this.socket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        this.socket.onclose = () => {
          console.log('WebSocket connection closed');
          this.isConnected = false;
          this.onConnectionChange?.(false);
          this.attemptReconnect();
        };

        this.socket.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.onError?.(error);
          resolve(false);
        };
      } catch (error) {
        console.error('Failed to create WebSocket connection:', error);
        this.onError?.(error);
        resolve(false);
      }
    });
  }

  private handleMessage(data: any) {
    switch (data.type) {
      case 'facial_analysis':
        this.onFacialAnalysis?.(data.data);
        break;
      case 'voice_analysis':
        this.onVoiceAnalysis?.(data.data);
        break;
      case 'question_generated':
        this.onQuestionGenerated?.(data);
        break;
      case 'response_received':
        this.onResponseReceived?.(data);
        break;
      case 'heartbeat_response':
        // Keep-alive response
        break;
      case 'error':
        console.error('WebSocket error from server:', data.error);
        this.onError?.(data);
        break;
      default:
        console.log('Unknown message type:', data.type);
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      
      setTimeout(() => {
        if (this.sessionId) {
          this.connect(this.sessionId);
        }
      }, this.reconnectInterval);
    } else {
      console.error('Max reconnection attempts reached');
      this.onError?.({ type: 'max_reconnect_reached' });
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this.isConnected = false;
      this.sessionId = '';
      this.reconnectAttempts = 0;
    }
  }

  sendFacialData(imageData: string) {
    if (this.socket && this.isConnected) {
      const message = {
        type: 'facial',
        data: {
          image: imageData
        }
      };
      this.socket.send(JSON.stringify(message));
    }
  }

  sendVoiceData(audioData: string) {
    if (this.socket && this.isConnected) {
      const message = {
        type: 'voice',
        data: {
          audio: audioData
        }
      };
      this.socket.send(JSON.stringify(message));
    }
  }

  sendHeartbeat() {
    if (this.socket && this.isConnected) {
      const message = {
        type: 'heartbeat',
        data: {}
      };
      this.socket.send(JSON.stringify(message));
    }
  }

  requestQuestion(data: any) {
    if (this.socket && this.isConnected) {
      // For WebSocket connection to backend node.js, we need to emit events
      const message = {
        type: 'request-question',
        data: data
      };
      this.socket.send(JSON.stringify(message));
    }
  }

  submitResponse(data: any) {
    if (this.socket && this.isConnected) {
      const message = {
        type: 'submit-response',
        data: data
      };
      this.socket.send(JSON.stringify(message));
    }
  }

  // Event handler setters
  setOnFacialAnalysis(handler: (data: any) => void) {
    this.onFacialAnalysis = handler;
  }

  setOnVoiceAnalysis(handler: (data: any) => void) {
    this.onVoiceAnalysis = handler;
  }

  setOnConnectionChange(handler: (isConnected: boolean) => void) {
    this.onConnectionChange = handler;
  }

  setOnError(handler: (error: any) => void) {
    this.onError = handler;
  }

  setOnQuestionGenerated(handler: (data: any) => void) {
    this.onQuestionGenerated = handler;
  }

  setOnResponseReceived(handler: (data: any) => void) {
    this.onResponseReceived = handler;
  }

  getConnectionStatus(): boolean {
    return this.isConnected;
  }

  getSessionId(): string {
    return this.sessionId;
  }
}

// Singleton instance
export const webSocketService = new WebSocketService();
export default webSocketService;