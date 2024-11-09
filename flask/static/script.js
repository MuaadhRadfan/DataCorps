const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
const sidebar = document.querySelector('.sidebar');

mobileMenuBtn.addEventListener('click', () => {
    sidebar.classList.toggle('active');
});

document.addEventListener('click', (e) => {
    if (window.innerWidth <= 768 && 
        !e.target.closest('.sidebar') && 
        !e.target.closest('.mobile-menu-btn')) {
        sidebar.classList.remove('active');
    }
});