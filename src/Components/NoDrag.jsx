import { useEffect, useRef } from 'react';

// react-tinder-card starts a drag from native mousedown/touchstart listeners on the card,
// which fire before React's delegated handlers. Stop them natively so buttons and panels
// inside a card can be clicked and scrolled without swiping it.
const NoDrag = ({ children, className, as: Tag = 'div' }) => {
    const ref = useRef(null);
    useEffect(() => {
        const el = ref.current;
        const stop = (e) => e.stopPropagation();
        el.addEventListener('mousedown', stop);
        el.addEventListener('touchstart', stop, { passive: true });
        return () => {
            el.removeEventListener('mousedown', stop);
            el.removeEventListener('touchstart', stop);
        };
    }, []);
    return <Tag ref={ref} className={className}>{children}</Tag>;
};

export default NoDrag;
