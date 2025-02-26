// script.js

// Handle Registration
document.getElementById("registerForm")?.addEventListener("submit", function(event) {
    event.preventDefault();
  
    const newUsername = document.getElementById("newUsername").value;
    const newPassword = document.getElementById("newPassword").value;
    const registerMessage = document.getElementById("register-message");
  
    // Check if username already exists in localStorage
    if (localStorage.getItem(newUsername)) {
      registerMessage.style.color = "red";
      registerMessage.textContent = "Username already exists!";
    } else {
      // Store the new user in localStorage
      localStorage.setItem(newUsername, newPassword);
      registerMessage.style.color = "green";
      registerMessage.textContent = "Registration successful! Redirecting to login...";
  
      // Redirect to login page after 2 seconds
      setTimeout(() => {
        window.location.href = "login.html";
      }, 2000);
    }
  });
  
  
  // Handle Login
  document.getElementById("loginForm")?.addEventListener("submit", function(event) {
    event.preventDefault();
  
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;
    const errorMessage = document.getElementById("error-message");
  
    // Retrieve stored password from localStorage
    const storedPassword = localStorage.getItem(username);
  
    if (storedPassword && storedPassword === password) {
      errorMessage.style.color = "green";
      errorMessage.textContent = "Login successful!";
      setTimeout(() => {
        window.location.href = "webpage.html";
      }, 1000);
    } else {
      errorMessage.style.color = "red";
      errorMessage.textContent = "Invalid username or password.";
    }
  });
  