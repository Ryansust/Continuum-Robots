function contact_pt = local_get_disk_position(P, disk_id)
% ==========================================================
% Convert physical disk index (1~13) to plotting position
%
% Marker locations:
%   Marker1 -> Disk1
%   Marker2 -> Disk3
%   Marker3 -> Disk5
%   Marker4 -> Disk7
%   Marker5 -> Disk9
%   Marker6 -> Disk11
%   Marker7 -> Disk13
%
% Even-numbered disks are interpolated between adjacent markers.
% ==========================================================

switch disk_id

    case 13
        contact_pt = P(:,1);

    case 12
        contact_pt = 0.5*(P(:,1)+P(:,2));

    case 11
        contact_pt = P(:,2);

    case 10
        contact_pt = 0.5*(P(:,2)+P(:,3));

    case 9
        contact_pt = P(:,3);

    case 8
        contact_pt = 0.5*(P(:,3)+P(:,4));

    case 7
        contact_pt = P(:,4);

    case 6
        contact_pt = 0.5*(P(:,4)+P(:,5));

    case 5
        contact_pt = P(:,5);

    case 4
        contact_pt = 0.5*(P(:,5)+P(:,6));

    case 3
        contact_pt = P(:,6);

    case 2
        contact_pt = 0.5*(P(:,6)+P(:,7));

    case 1
        contact_pt = P(:,7);

    otherwise
        error('Disk index must be between 1 and 13.');

end

end