import { io, Socket } from 'socket.io-client';

class SocketIOService {
  private socket: Socket | null = null;
  private sessionId: string = '';
  private isConnected: boolean = false;

  // Event handlers
  private onQuestionGenerated: ((data: any) => void) | null = null;
  private onResponseReceived: ((data: any) => void) | null = null;
  private onConnectionChange: ((isConnected: boolean) => void) | null = null;
  private onError: ((error: any) => void) | null = null;

  connect(sessionId: string): Promise<boolean> {
    this.sessionId = sessionId;
    
    return new Promise((resolve) => {
      try {
        // Connect to the Node.js backend
        this.socket = io('http://localhost:5000', {
          transports: ['websocket', 'polling'],
          forceNew: true
        });

        this.socket.on('connect', () => {
          console.log('Socket.IO connected to backend');
          this.isConnected = true;
          this.onConnectionChange?.(true);
          
          // Join the session
          this.socket!.emit('join-session', sessionId);
          resolve(true);
        });

        this.socket.on('session-joined', (data) => {
          console.log('Successfully joined session:', data);
        });

        this.socket.on('question-generated', (data) => {
          console.log('Question generated:', data);
          this.onQuestionGenerated?.(data);
        });

        this.socket.on('response-received', (data) => {
          console.log('Response received:', data);
          this.onResponseReceived?.(data);
        });

        this.socket.on('disconnect', () => {
          console.log('Socket.IO disconnected');
          this.isConnected = false;
          this.onConnectionChange?.(false);
        });

        this.socket.on('error', (error) => {
          console.error('Socket.IO error:', error);
          this.onError?.(error);
          resolve(false);
        });

        this.socket.on('connect_error', (error) => {
          console.error('Socket.IO connection error:', error);
          this.onError?.(error);
          resolve(false);
        });

      } catch (error) {
        console.error('Failed to create Socket.IO connection:', error);
        this.onError?.(error);
        resolve(false);
      }
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.isConnected = false;
      this.sessionId = '';
    }
  }

  requestQuestion(data: {
    sessionId: string;
    category?: string;
    difficulty?: string;
    jobRole: string;
    resumeData?: any;
  }) {
    if (this.socket && this.isConnected) {
      console.log('Requesting question with data:', data);
      this.socket.emit('request-question', data);
    } else {
      console.error('Socket not connected, cannot request question');
    }
  }

  submitResponse(data: {
    sessionId: string;
    questionId: string;
    response: string;
    duration: number;
    question: string;
  }) {
    if (this.socket && this.isConnected) {
      console.log('Submitting response:', data);
      this.socket.emit('submit-response', data);
    } else {
      console.error('Socket not connected, cannot submit response');
    }
  }

  // Event handler setters
  setOnQuestionGenerated(handler: (data: any) => void) {
    this.onQuestionGenerated = handler;
  }

  setOnResponseReceived(handler: (data: any) => void) {
    this.onResponseReceived = handler;
  }

  setOnConnectionChange(handler: (isConnected: boolean) => void) {
    this.onConnectionChange = handler;
  }

  setOnError(handler: (error: any) => void) {
    this.onError = handler;
  }

  getConnectionStatus(): boolean {
    return this.isConnected;
  }

  getSessionId(): string {
    return this.sessionId;
  }
}

// Singleton instance
export const socketIOService = new SocketIOService();
export default socketIOService;