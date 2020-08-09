import React from 'react';
import { BrowserRouter } from "react-router-dom";
import './App.css';

import BaseRouter from './routes';

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <BaseRouter/>
      </BrowserRouter>
    </div>
  );
}

export default App;
