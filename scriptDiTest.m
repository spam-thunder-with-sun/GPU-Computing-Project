clauses_ = [1 0 0 0 0 1; 1 1 1 0 0 0];
%[1 0 -1; 1 1 1];
%[1 0 0 0 0 1; 1 1 1 0 0 0];
clauses = clauses_';
possible = [    
    0 0 0 1 1 1; %no
    1 0 0 0 1 1; %si
    0 1 0 1 0 1; %si
    1 1 0 0 0 1; %si
    0 0 1 1 1 0; %no
    1 0 1 0 1 0; %si
    0 1 1 1 0 0; %no
    1 1 1 0 0 0;]; %si

disp(possible * clauses);

%disp(foo1);
%disp(foo2);
%disp(foo1 * foo2);


%{
[
    0 0 0 1 1 1; %no
    1 0 0 0 1 1; %si
    0 1 0 1 0 1; %si
    1 1 0 0 0 1; %si
    0 0 1 1 1 0; %no
    1 0 1 0 1 0; %si
    0 1 1 1 0 0; %no
    1 1 1 0 0 0;]; %si

    -1 -1 -1; %no
    1 -1 -1; %si
    -1 1 -1; %si
    1 1 -1; %si
    -1 -1 1; %no
    1 -1 1; %si
    -1 1 1; %no
    1 1 1]; %si

    0 0 0; %no
    1 0 0; %si
    0 1 0; %si
    1 1 0; %si
    0 0 1; %no
    1 0 1; %si
    0 1 1; %no
    1 1 1]; %si
%}











