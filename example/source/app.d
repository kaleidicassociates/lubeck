import mir.ndslice.slice: sliced;
import lubeck: mldivide;

// Solves AX=B
int main()
{
    auto a = [
         1, -1,  1,
         2,  2, -4,
        -1,  5,  0].sliced(3, 3);
    auto b = [
         2.0,  0,
        -6  , -6,
         9  ,  1].sliced(3, 2);
    auto t = [
         1.0, -1,
         2  ,  0,
         3  ,  1].sliced(3, 2);

    auto x = a.mldivide(b);

    // Check result
    if(x != t)
        return 1;

    // OK
    return 0;
}
