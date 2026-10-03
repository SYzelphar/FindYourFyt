// The signed-in user shown in the navbar and on the profile page.
export const USER = { name: 'Shlok Salgaonkar' };

export const initials = (name) =>
    name.split(' ').filter(Boolean).map((part) => part[0].toUpperCase()).slice(0, 2).join('');
