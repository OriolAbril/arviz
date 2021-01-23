(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("21b6ea09-d228-46b7-ad5c-dbbeb7636383");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '21b6ea09-d228-46b7-ad5c-dbbeb7636383' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js": "T2yuo9Oe71Cz/I4X9Ac5+gpEa5a8PpJCDlqKYO0CfAuEszu1JrXLl8YugMqYe3sM", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js": "98GDGJ0kOMCUMUePhksaQ/GYgB3+NH9h996V88sh3aOiUNX3N+fLXAtry6xctSZ6", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js": "89bArO+nlbP3sgakeHjCo1JYxYR5wufVgA3IbUvDY+K7w4zyxJqssu7wVnfeKCq8"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"3a2d5616-1ff3-47d7-8e99-4f4de958f81a":{"roots":{"references":[{"attributes":{},"id":"21880","type":"UnionRenderers"},{"attributes":{"below":[{"id":"21769"}],"center":[{"id":"21772"},{"id":"21776"}],"left":[{"id":"21773"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"21835"},{"id":"21837"},{"id":"21838"},{"id":"21839"},{"id":"21842"}],"title":{"id":"21844"},"toolbar":{"id":"21787"},"toolbar_location":null,"x_range":{"id":"21761"},"x_scale":{"id":"21765"},"y_range":{"id":"21763"},"y_scale":{"id":"21767"}},"id":"21760","subtype":"Figure","type":"Plot"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"21785","type":"BoxAnnotation"},{"attributes":{},"id":"21864","type":"BasicTickFormatter"},{"attributes":{},"id":"21881","type":"Selection"},{"attributes":{},"id":"21882","type":"UnionRenderers"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"21786","type":"PolyAnnotation"},{"attributes":{"data":{"rug_x":{"__ndarray__":"jQwCEA1Gsz9bBMLb9PjIP1Qd9Zram7E/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/aF85wG2piz9oXznAbamLP2hfOcBtqYs/aF85wG2piz9oXznAbamLP2hfOcBtqYs/pI3yRkqEyT9c/+ob+nG6P4OiBeyjALU/mx4fY+a33D9wF2c1cbTRPy51Aws2htg/P07TEgOYwz9FOncTGHy5P/BQ+ANPucc/uRO6PYJJzj9CRCWTDYpmP+UvVnGFsrI/8JR71fNwcj8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"21869"},"selection_policy":{"id":"21870"}},"id":"21841","type":"ColumnDataSource"},{"attributes":{},"id":"21867","type":"Selection"},{"attributes":{"source":{"id":"21841"}},"id":"21843","type":"CDSView"},{"attributes":{},"id":"21868","type":"UnionRenderers"},{"attributes":{"source":{"id":"21832"}},"id":"21836","type":"CDSView"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"21840","type":"Dash"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2515582690238702},"id":"21837","type":"Span"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.1782444431478369},"id":"21838","type":"Span"},{"attributes":{"data_source":{"id":"21841"},"glyph":{"id":"21840"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"21843"}},"id":"21842","type":"GlyphRenderer"},{"attributes":{"text":"tau"},"id":"21844","type":"Title"},{"attributes":{"source":{"id":"21846"}},"id":"21850","type":"CDSView"},{"attributes":{},"id":"21869","type":"Selection"},{"attributes":{},"id":"21870","type":"UnionRenderers"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"21839","type":"Span"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"TOY7mm62yD/upazjteTNP/DqXfy8Lcs/HHCIRHVOzT9orzckCyTOP6y8CkfbP8w/iNx/9NF5yz+IfM+LFJ/NPygi8KloJ8w/oK0q3zNVzD/YcLH58jHPP6hWLqhoMNQ/IGaMl5nu0j8AebTCVJLRP+D0hUiIM80/uNpKBpu90z/A00Fq0J3TP2iCFjW8ldY/kLBgWUSo1j8QCPgZgPnXPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"21867"},"selection_policy":{"id":"21868"}},"id":"21832","type":"ColumnDataSource"},{"attributes":{"data":{"rug_x":{"__ndarray__":"fV36E1z/6j89DycBWWfXP73VlTJ7YsE/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/Oqay45Jr6D86SHRMZcflP7HThSU1z+I/VdSaCTtd6D9Hvab2ZmSwP8UA5kQ6d4M/0GULqag1oz+aI/Yi4T7rP55DEK8H/NA/qgGd6qjX1D+VqnS/h2ThP/UnuP7VN+Q/TGXHJdeQ2z8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"21881"},"selection_policy":{"id":"21882"}},"id":"21855","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21848","type":"Circle"},{"attributes":{"children":[{"id":"21886"},{"id":"21884"}]},"id":"21887","type":"Column"},{"attributes":{},"id":"21879","type":"Selection"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21847","type":"Circle"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"bcJe+Pxh3D88Eu5SUq3NPyL+eQAns8s/mzeOVbYJ1z+YvNPsmzTWP87j+UyletE/sIr1s8Bnzz8YN/msnr7PP8gPs0h4ec8/RNGUEkZ90z8g12riYrDUP+AlTxYjYc4/UBTOvdhAzD+AFwpyJ0DOPxDSzcUXbc4/kDNyJsikyj9AY0p3Si3PP8AWdsF70MQ/4I8dAXxLyD+grhW5nZrSPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"21879"},"selection_policy":{"id":"21880"}},"id":"21846","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21834","type":"Circle"},{"attributes":{"toolbars":[{"id":"21787"},{"id":"21823"}],"tools":[{"id":"21777"},{"id":"21778"},{"id":"21779"},{"id":"21780"},{"id":"21781"},{"id":"21782"},{"id":"21783"},{"id":"21784"},{"id":"21813"},{"id":"21814"},{"id":"21815"},{"id":"21816"},{"id":"21817"},{"id":"21818"},{"id":"21819"},{"id":"21820"}]},"id":"21885","type":"ProxyToolbar"},{"attributes":{"data_source":{"id":"21846"},"glyph":{"id":"21847"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"21848"},"selection_glyph":null,"view":{"id":"21850"}},"id":"21849","type":"GlyphRenderer"},{"attributes":{"source":{"id":"21855"}},"id":"21857","type":"CDSView"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"21854","type":"Dash"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21833","type":"Circle"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"21876"},"ticker":{"id":"21806"}},"id":"21805","type":"LinearAxis"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2148430013731262},"id":"21851","type":"Span"},{"attributes":{"below":[{"id":"21805"}],"center":[{"id":"21808"},{"id":"21812"}],"left":[{"id":"21809"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"21849"},{"id":"21851"},{"id":"21852"},{"id":"21853"},{"id":"21856"}],"title":{"id":"21858"},"toolbar":{"id":"21823"},"toolbar_location":null,"x_range":{"id":"21797"},"x_scale":{"id":"21801"},"y_range":{"id":"21799"},"y_scale":{"id":"21803"}},"id":"21796","subtype":"Figure","type":"Plot"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.15209716424958658},"id":"21852","type":"Span"},{"attributes":{"end":1,"start":-0.05},"id":"21799","type":"DataRange1d"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"21874"},"ticker":{"id":"21810"}},"id":"21809","type":"LinearAxis"},{"attributes":{"data_source":{"id":"21855"},"glyph":{"id":"21854"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"21857"}},"id":"21856","type":"GlyphRenderer"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"21862"},"ticker":{"id":"21774"}},"id":"21773","type":"LinearAxis"},{"attributes":{"text":"mu"},"id":"21858","type":"Title"},{"attributes":{},"id":"21801","type":"LinearScale"},{"attributes":{},"id":"21803","type":"LinearScale"},{"attributes":{"data_source":{"id":"21832"},"glyph":{"id":"21833"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"21834"},"selection_glyph":null,"view":{"id":"21836"}},"id":"21835","type":"GlyphRenderer"},{"attributes":{},"id":"21761","type":"DataRange1d"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"21853","type":"Span"},{"attributes":{},"id":"21806","type":"BasicTicker"},{"attributes":{"axis":{"id":"21805"},"ticker":null},"id":"21808","type":"Grid"},{"attributes":{},"id":"21765","type":"LinearScale"},{"attributes":{"callback":null},"id":"21784","type":"HoverTool"},{"attributes":{"toolbar":{"id":"21885"},"toolbar_location":"above"},"id":"21886","type":"ToolbarBox"},{"attributes":{"end":1,"start":-0.05},"id":"21763","type":"DataRange1d"},{"attributes":{},"id":"21810","type":"BasicTicker"},{"attributes":{},"id":"21774","type":"BasicTicker"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"21864"},"ticker":{"id":"21770"}},"id":"21769","type":"LinearAxis"},{"attributes":{},"id":"21874","type":"BasicTickFormatter"},{"attributes":{},"id":"21767","type":"LinearScale"},{"attributes":{},"id":"21770","type":"BasicTicker"},{"attributes":{},"id":"21777","type":"ResetTool"},{"attributes":{"axis":{"id":"21769"},"ticker":null},"id":"21772","type":"Grid"},{"attributes":{"axis":{"id":"21773"},"dimension":1,"ticker":null},"id":"21776","type":"Grid"},{"attributes":{},"id":"21862","type":"BasicTickFormatter"},{"attributes":{},"id":"21797","type":"DataRange1d"},{"attributes":{"overlay":{"id":"21785"}},"id":"21779","type":"BoxZoomTool"},{"attributes":{},"id":"21778","type":"PanTool"},{"attributes":{},"id":"21780","type":"WheelZoomTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"21821","type":"BoxAnnotation"},{"attributes":{},"id":"21783","type":"SaveTool"},{"attributes":{"overlay":{"id":"21786"}},"id":"21781","type":"LassoSelectTool"},{"attributes":{"children":[[{"id":"21760"},0,0],[{"id":"21796"},0,1]]},"id":"21884","type":"GridBox"},{"attributes":{"overlay":{"id":"21822"}},"id":"21817","type":"LassoSelectTool"},{"attributes":{},"id":"21782","type":"UndoTool"},{"attributes":{},"id":"21876","type":"BasicTickFormatter"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"21777"},{"id":"21778"},{"id":"21779"},{"id":"21780"},{"id":"21781"},{"id":"21782"},{"id":"21783"},{"id":"21784"}]},"id":"21787","type":"Toolbar"},{"attributes":{"overlay":{"id":"21821"}},"id":"21815","type":"BoxZoomTool"},{"attributes":{},"id":"21819","type":"SaveTool"},{"attributes":{},"id":"21814","type":"PanTool"},{"attributes":{},"id":"21816","type":"WheelZoomTool"},{"attributes":{},"id":"21818","type":"UndoTool"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"21822","type":"PolyAnnotation"},{"attributes":{},"id":"21813","type":"ResetTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"21813"},{"id":"21814"},{"id":"21815"},{"id":"21816"},{"id":"21817"},{"id":"21818"},{"id":"21819"},{"id":"21820"}]},"id":"21823","type":"Toolbar"},{"attributes":{"axis":{"id":"21809"},"dimension":1,"ticker":null},"id":"21812","type":"Grid"},{"attributes":{"callback":null},"id":"21820","type":"HoverTool"}],"root_ids":["21887"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"3a2d5616-1ff3-47d7-8e99-4f4de958f81a","root_ids":["21887"],"roots":{"21887":"21b6ea09-d228-46b7-ad5c-dbbeb7636383"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();